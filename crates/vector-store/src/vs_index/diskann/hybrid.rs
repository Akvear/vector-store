/*
 * Copyright 2026-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

//! A DiskANN backend that keeps the graph in RAM but not the vectors.
//!
//! The vectors we index already live in ScyllaDB, so the default
//! [`InmemBackend`](super::inmem::InmemBackend) stores every one of them a
//! second time. This backend removes that copy: adjacency lists, the external
//! id mapping and slot bookkeeping stay in RAM (delegated to
//! [`diskann_inmem`]), while vector data is fetched on demand through
//! [`VectorReader`].
//!
//! # How the delegation works
//!
//! [`HybridProvider`] wraps an `InmemProvider<NoVectors, PrimaryId>`. The
//! [`NoVectors`] layer reports a one-byte record, so inmem allocates slots and
//! maintains its id map exactly as before but stores no vector payload. All of
//! `DataProvider`, `Delete` and the slot half of `SetElement` are then plain
//! delegation.
//!
//! Only the two hooks that actually read vector data are ours:
//!
//! * [`glue::SearchAccessor::expand_beam`] — one batched fetch per search round.
//! * [`glue::PruneAccessor::fill`] — one batched fetch per prune round.
//!
//! # What still lives in RAM
//!
//! Three things, none of which grow with the dataset:
//!
//! * The graph's start point. It is handed to inmem as raw data with no
//!   external id, so it has no primary key and can never be fetched back.
//! * Vectors of in-flight inserts. Insert prunes edges using the new vector
//!   before its row is reliably readable, so it rides along until the insert
//!   completes. [`InflightGuard`] removes it on both success and failure.
//! * Per-operation working sets: candidates fetched for one prune, dropped
//!   when it ends.

use super::DiskannBackend;
use super::DiskannParams;
use crate::PrimaryId;
use anyhow::Context as _;
use anyhow::anyhow;
use anyhow::bail;
use diskann::ANNError;
use diskann::ANNResult;
use diskann::default_post_processor;
use diskann::graph::AdjacencyList;
use diskann::graph::DiskANNIndex;
use diskann::graph::SearchOutputBuffer;
use diskann::graph::glue;
use diskann::graph::workingset;
use diskann::neighbor::Neighbor;
use diskann::provider;
use diskann::provider::DataProvider;
use diskann::provider::Delete;
use diskann::provider::ElementStatus;
use diskann::provider::HasId;
use diskann::provider::NeighborAccessor as _;
use diskann::provider::SetElement;
use diskann::utils::VectorRepr;
use diskann_inmem::Context as InmemContext;
use diskann_inmem::Provider as InmemProvider;
use diskann_inmem::Strategy as InmemStrategy;
use diskann_inmem::layers;
use diskann_inmem::num::Bytes;
use diskann_inmem::provider::Config as InmemProviderConfig;
use diskann_inmem::provider::PruneAccessor as InmemPruneAccessor;
use diskann_vector::PreprocessedDistanceFunction as _;
use diskann_vector::distance::Metric;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
use std::sync::RwLock;

#[derive(Clone)]
pub(super) struct HybridBackend {
    strategy: HybridStrategy,
    context: InmemContext,
}

impl HybridBackend {
    #[allow(dead_code, reason = "wired into DiskannIndexFactory in a follow-up")]
    pub(super) fn new() -> Self {
        Self {
            strategy: HybridStrategy,
            context: InmemContext,
        }
    }
}

impl DiskannBackend for HybridBackend {
    type Provider = HybridProvider;
    type Strategy = HybridStrategy;

    fn create_index(
        &self,
        params: &DiskannParams,
        start_point: &[f32],
    ) -> anyhow::Result<DiskANNIndex<Self::Provider>> {
        let provider = HybridProvider::new(
            start_point,
            usize::from(params.dim.0),
            params.metric,
            usize::from(params.max_points),
            params.config.max_degree().get(),
        )
        .context("failed to create HybridProvider")?;

        Ok(DiskANNIndex::new(params.config.clone(), provider, None))
    }

    fn strategy(&self) -> &Self::Strategy {
        &self.strategy
    }

    fn context(&self) -> &InmemContext {
        &self.context
    }
}

/// A [`layers::Layer`] that stores no vector payload.
#[derive(Debug)]
pub(super) struct NoVectors {
    dim: usize,
}

impl NoVectors {
    pub(super) fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl layers::Layer for NoVectors {
    fn bytes(&self) -> Bytes {
        // `InmemProvider::new` writes the start points through
        // `Matrix::row_iter_mut()`, which is `chunks_exact_mut(ncols)` — and
        // `chunks_exact_mut(0)` panics unconditionally. One byte avoids that.
        //
        // For comparison, `Full<f32>` at 768 dims occupies 3088 bytes per slot.
        Bytes::new(1)
    }
}

impl layers::Set<&[f32]> for NoVectors {
    /// Validate the dimension and discard the data.
    fn set(&self, element: &[f32], _bytes: &mut [u8]) -> ANNResult<()> {
        if element.len() != self.dim {
            return Err(ANNError::message(format!(
                "wrong dimension: got {}, expected {}",
                element.len(),
                self.dim
            )));
        }
        Ok(())
    }
}

/// Stand-in for the layer-level distance function.
///
/// `diskann_inmem`'s `PruneStrategy` impl is bounded on
/// `L: Layer + AsDistance`, and that is the only public way to reach inmem's
/// adjacency store. We never route distance computations through it — pruning
/// uses [`HybridPruneAccessor`] — so this exists purely to satisfy the bound
/// and errors if it is ever reached.
#[derive(Debug)]
struct NoDistance;

static NO_DISTANCE: NoDistance = NoDistance;

impl layers::Distance for NoDistance {
    fn evaluate(&self, _x: &[u8], _y: &[u8]) -> ANNResult<f32> {
        Err(ANNError::message(
            "the NoVectors layer stores no vector data; \
             distances must go through HybridPruneAccessor",
        ))
    }
}

impl layers::AsDistance for NoVectors {
    fn as_distance(&self) -> &dyn layers::Distance {
        &NO_DISTANCE
    }
}

/// Vectors held in RAM only for the duration of an insert.
type Inflight = Arc<RwLock<BTreeMap<u32, Box<[f32]>>>>;

/// A DiskANN provider whose vector reads go to [`VectorReader`].
#[derive(Debug)]
pub(super) struct HybridProvider {
    inner: InmemProvider<NoVectors, PrimaryId>,
    inflight: Inflight,
    start_id: u32,
    start_vector: Box<[f32]>,
    dim: usize,
    metric: Metric,
}

impl HybridProvider {
    /// Build a provider whose graph holds up to `max_points` vectors.
    ///
    /// `start_point` becomes the graph's sole start point and is retained in
    /// RAM, because it is registered with inmem as raw data and therefore has
    /// no primary key to fetch it by.
    pub(super) fn new(
        start_point: &[f32],
        dim: usize,
        metric: Metric,
        max_points: usize,
        max_degree: usize,
    ) -> anyhow::Result<Self> {
        if start_point.len() != dim {
            bail!(
                "start point has dimension {} but the index expects {dim}",
                start_point.len()
            );
        }

        let inner = InmemProvider::new(
            NoVectors::new(dim),
            InmemProviderConfig::new(max_points, max_degree),
            [start_point],
        )
        .context("failed to create the inner in-memory provider")?;

        let start_id = u32::try_from(max_points)
            .with_context(|| format!("max_points {max_points} does not fit in a u32 slot id"))?;

        verify_start_slot(&inner, start_id)?;

        Ok(Self {
            inner,
            inflight: Inflight::default(),
            start_id,
            start_vector: start_point.into(),
            dim,
            metric,
        })
    }

    /// Resolve internal `ids` to vector data with a single call to ScyllaDB.
    ///
    /// The result is index-aligned with `ids`. `None` means "unavailable, skip
    /// this candidate": the id has no external mapping (already deleted) or the
    /// source did not return a row for it. Both are tolerated by
    /// `expand_beam` and `fill`.
    async fn load(&self, ids: &[u32]) -> ANNResult<Vec<Option<Box<[f32]>>>> {
        let mut out: Vec<Option<Box<[f32]>>> = Vec::with_capacity(ids.len());

        // Positions in `out` that the source has to fill, and the external ids
        // to ask it for. Kept aligned with each other.
        let mut slots: Vec<usize> = Vec::new();
        let mut wanted: Vec<PrimaryId> = Vec::new();

        let inflight = self.inflight.read().unwrap();
        for (position, &id) in ids.iter().enumerate() {
            if id == self.start_id {
                out.push(Some(self.start_vector.clone()));
                continue;
            }

            if let Some(vector) = inflight.get(&id) {
                out.push(Some(vector.clone()));
                continue;
            }

            out.push(None);
            if let Ok(external) = self.inner.to_external_id(&InmemContext, id) {
                slots.push(position);
                wanted.push(external);
            }
        }
        drop(inflight);

        if wanted.is_empty() {
            return Ok(out);
        }

        let fetched = vec![Some(vec![0f32].into_boxed_slice()); wanted.len()];

        if fetched.len() != wanted.len() {
            return Err(ANNError::message(format!(
                "vector source returned {} results for {} ids; \
                 implementations must return one entry per id, in order",
                fetched.len(),
                wanted.len()
            )));
        }

        for (position, vector) in slots.into_iter().zip(fetched) {
            out[position] = vector;
        }

        Ok(out)
    }

    /// Borrow an accessor over the inner provider's adjacency store.
    ///
    /// inmem keeps its adjacency lists private; its `PruneAccessor` is the only
    /// public handle, and it implements both `NeighborAccessor` and
    /// `NeighborAccessorMut`.
    fn adjacency(&self) -> ANNResult<InmemPruneAccessor<'_>> {
        glue::PruneStrategy::prune_accessor(&InmemStrategy, &self.inner, &InmemContext, 0)
    }
}

/// Check that the start point really landed in the slot we expect.
///
/// `diskann-inmem` lays writable slots out over `[0, max_points)` and frozen
/// (start) points over `[max_points, max_points + n)`, but `Store::frozen()` is
/// `pub(crate)`, so the id is not directly observable. `status_by_internal_id`
/// *is* public and distinguishes the two ranges: an unallocated writable slot
/// reports `Deleted` while a frozen slot reports `Valid`.
fn verify_start_slot(
    inner: &InmemProvider<NoVectors, PrimaryId>,
    start_id: u32,
) -> anyhow::Result<()> {
    // These futures are synchronous in `diskann-inmem`; `now_or_never` asserts
    // that rather than risking a block inside the actor.
    let status = |id: u32| -> anyhow::Result<ElementStatus> {
        futures::FutureExt::now_or_never(inner.status_by_internal_id(&InmemContext, id))
            .ok_or_else(|| anyhow!("diskann-inmem status probe for slot {id} unexpectedly pended"))?
            .map_err(|err| anyhow!("status probe for slot {id} failed: {err:?}"))
    };

    let at_start = status(start_id)?;
    if at_start != ElementStatus::Valid {
        bail!(
            "expected the start point in slot {start_id}, found {at_start:?}; \
             the diskann-inmem frozen slot layout has probably changed"
        );
    }

    if let Some(below) = start_id.checked_sub(1) {
        let at_below = status(below)?;
        if at_below != ElementStatus::Deleted {
            bail!(
                "expected slot {below} to be an unallocated writable slot, found {at_below:?}; \
                 the diskann-inmem frozen slot layout has probably changed"
            );
        }
    }

    Ok(())
}

impl DataProvider for HybridProvider {
    type Context = InmemContext;
    type InternalId = u32;
    type ExternalId = PrimaryId;
    type Error = ANNError;
    type Guard = InflightGuard;

    fn to_internal_id(
        &self,
        context: &Self::Context,
        gid: &Self::ExternalId,
    ) -> Result<Self::InternalId, Self::Error> {
        self.inner.to_internal_id(context, gid)
    }

    fn to_external_id(
        &self,
        context: &Self::Context,
        id: Self::InternalId,
    ) -> Result<Self::ExternalId, Self::Error> {
        self.inner.to_external_id(context, id)
    }
}

impl SetElement<&[f32]> for HybridProvider {
    type SetError = ANNError;

    async fn set_element(
        &self,
        context: &Self::Context,
        id: &Self::ExternalId,
        element: &[f32],
    ) -> Result<Self::Guard, Self::SetError> {
        // The inner provider allocates the slot, registers the id mapping and
        // validates the dimension through `NoVectors::set`. No vector payload
        // is written.
        let guard = self.inner.set_element(context, id, element).await?;
        let internal = provider::Guard::id(&guard);

        // Insert prunes edges using this vector before the row is reliably
        // readable, so hold on to it until the insert finishes.
        self.inflight
            .write()
            .unwrap()
            .insert(internal, element.into());

        Ok(InflightGuard {
            inflight: Arc::clone(&self.inflight),
            id: internal,
        })
    }
}

impl Delete for HybridProvider {
    fn delete(
        &self,
        context: &Self::Context,
        gid: &Self::ExternalId,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send {
        // Retires the slot and drops the id mapping in one step, so
        // `to_internal_id` stops resolving the node.
        self.inner.delete(context, gid)
    }

    fn release(
        &self,
        _context: &Self::Context,
        _id: Self::InternalId,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send {
        // `delete` already reclaimed the slot, and the graph index never calls
        // this. Pure sync work, so no state machine is generated at all.
        std::future::ready(Ok(()))
    }

    fn status_by_internal_id(
        &self,
        context: &Self::Context,
        id: Self::InternalId,
    ) -> impl Future<Output = Result<ElementStatus, Self::Error>> + Send {
        self.inner.status_by_internal_id(context, id)
    }

    fn status_by_external_id(
        &self,
        context: &Self::Context,
        gid: &Self::ExternalId,
    ) -> impl Future<Output = Result<ElementStatus, Self::Error>> + Send {
        self.inner.status_by_external_id(context, gid)
    }
}

/// Drops the in-flight copy of a vector once its insert is over.
///
/// Both `complete` and `Drop` clear the entry, so a failed insert cleans up
/// too — `add_vector` only logs insert errors and moves on, which would make a
/// leak here silent and unbounded.
#[derive(Debug)]
pub(super) struct InflightGuard {
    inflight: Inflight,
    id: u32,
}

impl provider::Guard for InflightGuard {
    type Id = u32;

    async fn complete(self) {
        // The `Drop` trait does the work.
    }

    fn id(&self) -> Self::Id {
        self.id
    }
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        self.inflight.write().unwrap().remove(&self.id);
    }
}

/// Serves graph search: adjacency from the inner store, vectors from the source.
pub(super) struct HybridSearchAccessor<'a> {
    provider: &'a HybridProvider,
    adjacency: InmemPruneAccessor<'a>,
    distance: <f32 as VectorRepr>::QueryDistance,
    /// Reused across `expand_beam` calls.
    neighbors: AdjacencyList<u32>,
    /// Reused across `expand_beam` calls.
    candidates: Vec<u32>,
}

impl<'a> HybridSearchAccessor<'a> {
    fn new(provider: &'a HybridProvider, query: &[f32]) -> ANNResult<Self> {
        if query.len() != provider.dim {
            return Err(ANNError::message(format!(
                "query has dimension {} but the index expects {}",
                query.len(),
                provider.dim
            )));
        }

        Ok(Self {
            provider,
            adjacency: provider.adjacency()?,
            distance: f32::query_distance(query, provider.metric),
            neighbors: AdjacencyList::new(),
            candidates: Vec::new(),
        })
    }
}

impl HasId for HybridSearchAccessor<'_> {
    type Id = u32;
}

impl glue::SearchAccessor for HybridSearchAccessor<'_> {
    fn starting_points(&self) -> impl Future<Output = ANNResult<Vec<Self::Id>>> + Send {
        std::future::ready(Ok(vec![self.provider.start_id]))
    }

    async fn start_point_distances<F>(&mut self, mut f: F) -> ANNResult<()>
    where
        F: FnMut(Self::Id, f32) + Send,
    {
        // The start point is resident, so this never touches the source.
        let distance = self
            .distance
            .evaluate_similarity(&self.provider.start_vector[..]);
        f(self.provider.start_id, distance);
        Ok(())
    }

    async fn expand_beam<Itr, P, F>(
        &mut self,
        ids: Itr,
        mut pred: P,
        mut on_neighbors: F,
    ) -> ANNResult<()>
    where
        Itr: Iterator<Item = Self::Id> + Send,
        P: glue::HybridPredicate<Self::Id> + Send + Sync,
        F: FnMut(Self::Id, f32) + Send,
    {
        // Collect the whole round's survivors first, so the fetch below is one
        // request per round rather than one per candidate.
        let mut neighbors = std::mem::take(&mut self.neighbors);
        let mut candidates = std::mem::take(&mut self.candidates);
        candidates.clear();

        let collected = async {
            for id in ids {
                self.adjacency.get_neighbors(id, &mut neighbors).await?;
                // `eval_mut` is a test-and-set: it both filters already-scored
                // ids and claims these, which is what keeps `candidates` unique.
                candidates.extend(neighbors.iter().copied().filter(|i| pred.eval_mut(i)));
            }
            ANNResult::Ok(())
        }
        .await;

        self.neighbors = neighbors;
        let result = match collected {
            Err(err) => Err(err),
            Ok(()) => match self.provider.load(&candidates).await {
                Err(err) => Err(err),
                Ok(loaded) => {
                    for (&id, vector) in candidates.iter().zip(loaded) {
                        // A `None` is a row that vanished under us. Skipping is
                        // explicitly allowed and costs recall, not correctness.
                        if let Some(vector) = vector {
                            on_neighbors(id, self.distance.evaluate_similarity(&vector[..]));
                        }
                    }
                    Ok(())
                }
            },
        };

        self.candidates = candidates;
        result
    }
}

type WorkingSet = workingset::Map<u32, Box<[f32]>, workingset::map::Ref<[f32]>>;
type WorkingSetView<'a> = workingset::map::View<'a, u32, Box<[f32]>, workingset::map::Ref<[f32]>>;

/// Serves index construction: element-to-element distances over a candidate set.
pub(super) struct HybridPruneAccessor<'a> {
    provider: &'a HybridProvider,
    adjacency: InmemPruneAccessor<'a>,
    /// Doubles as a per-operation cache: the index reuses one accessor across
    /// every prune round of an insert or delete.
    set: WorkingSet,
    distance: <f32 as VectorRepr>::Distance,
}

impl<'a> HybridPruneAccessor<'a> {
    fn new(provider: &'a HybridProvider, capacity: usize) -> ANNResult<Self> {
        let set = workingset::map::Builder::new(workingset::map::Capacity::Default).build(capacity);

        Ok(Self {
            provider,
            adjacency: provider.adjacency()?,
            set,
            distance: f32::distance(provider.metric, Some(provider.dim)),
        })
    }
}

impl HasId for HybridPruneAccessor<'_> {
    type Id = u32;
}

impl<'p> glue::PruneAccessor for HybridPruneAccessor<'p> {
    type Neighbors<'a>
        = provider::Neighbors<'a, InmemPruneAccessor<'p>>
    where
        Self: 'a;

    type ElementRef<'a> = &'a [f32];

    type View<'a>
        = WorkingSetView<'a>
    where
        Self: 'a;

    type Distance<'a>
        = <f32 as VectorRepr>::Distance
    where
        Self: 'a;

    fn neighbors(&mut self) -> Self::Neighbors<'_> {
        provider::Neighbors(&mut self.adjacency)
    }

    async fn fill<Itr>(&mut self, itr: Itr) -> ANNResult<(Self::View<'_>, Self::Distance<'_>)>
    where
        Itr: ExactSizeIterator<Item = Self::Id> + Clone + Send + Sync,
    {
        // `Map::fill` takes a *synchronous* closure, so the fetch cannot happen
        // inside it. Work out the misses, fetch them in one request, then let
        // `fill` drain the results.
        //
        // This is safe under eviction: `Map::fill` pins everything in `itr`
        // before it evicts anything.
        let misses: Vec<u32> = itr
            .clone()
            .filter(|id| !self.set.contains_key(id))
            .collect();

        let mut fetched: HashMap<u32, Box<[f32]>> = if misses.is_empty() {
            HashMap::new()
        } else {
            let loaded = self.provider.load(&misses).await?;
            misses
                .iter()
                .copied()
                .zip(loaded)
                .filter_map(|(id, vector)| vector.map(|vector| (id, vector)))
                .collect()
        };

        // Read before `fill` borrows `self.set` for the view's lifetime.
        let distance = self.distance;
        let view = self
            .set
            .fill(itr, |id| ANNResult::Ok(fetched.remove(&id)))?;

        Ok((view, distance))
    }
}

/// The strategy tying [`HybridProvider`] to DiskANN's graph operations.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct HybridStrategy;

impl<'a> glue::SearchStrategy<'a, HybridProvider, &'a [f32]> for HybridStrategy {
    type SearchAccessorError = ANNError;
    type SearchAccessor = HybridSearchAccessor<'a>;

    fn search_accessor(
        &'a self,
        provider: &'a HybridProvider,
        _context: &'a InmemContext,
        query: &'a [f32],
    ) -> Result<Self::SearchAccessor, Self::SearchAccessorError> {
        HybridSearchAccessor::new(provider, query)
    }
}

impl<'a> glue::DefaultPostProcessor<'a, HybridProvider, &'a [f32], PrimaryId> for HybridStrategy {
    default_post_processor!(TranslateIds);
}

impl glue::PruneStrategy<HybridProvider> for HybridStrategy {
    type PruneAccessor<'a> = HybridPruneAccessor<'a>;
    type PruneAccessorError = ANNError;

    fn prune_accessor<'a>(
        &'a self,
        provider: &'a HybridProvider,
        _context: &'a InmemContext,
        capacity: usize,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError> {
        HybridPruneAccessor::new(provider, capacity)
    }
}

impl<'a> glue::InsertStrategy<'a, HybridProvider, &'a [f32]> for HybridStrategy {
    type PruneStrategy = Self;

    fn prune_strategy(&self) -> Self::PruneStrategy {
        *self
    }
}

/// In-place delete over full-precision vectors.
///
/// `VisitedAndTopK` searches *using* the deleted vector as the query, and by
/// the time we hear about a delete its row is already gone from the base table,
/// so [`Self::get_delete_element`] cannot serve it.
impl glue::InplaceDeleteStrategy<HybridProvider> for HybridStrategy {
    type DeleteElement<'a> = &'a [f32];
    type DeleteElementGuard = Box<[f32]>;
    type DeleteElementError = ANNError;
    type PruneStrategy = Self;
    type DeleteSearchAccessor<'a> = HybridSearchAccessor<'a>;
    type SearchPostProcessor = glue::CopyIds;
    type SearchStrategy = Self;

    fn prune_strategy(&self) -> Self::PruneStrategy {
        *self
    }

    fn search_strategy(&self) -> Self::SearchStrategy {
        *self
    }

    fn search_post_processor(&self) -> Self::SearchPostProcessor {
        glue::CopyIds
    }

    async fn get_delete_element<'a>(
        &'a self,
        _provider: &'a HybridProvider,
        _context: &'a InmemContext,
        id: u32,
    ) -> Result<Self::DeleteElementGuard, Self::DeleteElementError> {
        Err(ANNError::message(format!(
            "cannot supply the vector for internal id {id}: this provider retains no \
             vector data, and a deleted row is already gone from the base table by the \
             time the delete reaches us. Use InplaceDeleteMethod::OneHop or \
             TwoHopAndOneHop, neither of which reads the deleted vector."
        )))
    }
}

/// Turns internal slot ids into [`PrimaryId`]s for the caller.
///
/// `diskann-inmem` ships an equivalent (`Translate`), but its impl is bound to
/// inmem's own search accessor and reaches a private field.
/// This version only needs the public `DataProvider::to_external_id`.
///
/// Dropping ids with no mapping is also what removes the start point from
/// results — it has no external id — so no `FilterStartPoints` step is needed.
#[derive(Debug, Default)]
pub(super) struct TranslateIds;

impl<'a> glue::SearchPostProcess<HybridSearchAccessor<'a>, &'a [f32], PrimaryId> for TranslateIds {
    type Error = ANNError;

    fn post_process<I, B>(
        &self,
        accessor: &mut HybridSearchAccessor<'a>,
        _query: &'a [f32],
        candidates: I,
        output: &mut B,
    ) -> impl Future<Output = Result<usize, Self::Error>> + Send
    where
        I: Iterator<Item = Neighbor<u32>> + Send,
        B: SearchOutputBuffer<PrimaryId> + Send + ?Sized,
    {
        let provider = accessor.provider;
        let mut count = 0;

        for candidate in candidates {
            let Ok(external) = provider.to_external_id(&InmemContext, *candidate.id()) else {
                // No mapping: the start point, or a node deleted mid-search.
                continue;
            };

            if output.push(external, candidate.distance()).is_available() {
                count += 1;
            } else {
                break;
            }
        }

        std::future::ready(Ok(count))
    }
}
