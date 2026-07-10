/*
 * Copyright 2025-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.0
 */

use crate::Config;
use crate::Dimensions;
use crate::Distance;
use crate::IndexKey;
use crate::Limit;
use crate::PrimaryKey;
use crate::Quantization;
use crate::SpaceType;
use crate::Vector;
use crate::VsIndexFactory;
use crate::memory::Memory;
use crate::perf;
use crate::table::PartitionId;
use crate::table::PrimaryId;
use crate::table::Table;
use crate::table::TableSearch;
use crate::vs_index::actor::VsIndex;
use crate::vs_index::factory::VsIndexConfiguration;

use std::fs::{File, OpenOptions};
use std::io::Seek;
use std::io::SeekFrom;
use std::io::Write;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use tokio::sync::{mpsc, watch};
use tracing::{Instrument, debug, debug_span, warn};

use diskann::error::ErrorContext;
use diskann::graph::Config as DiskannConfig;
use diskann::graph::config::defaults::ALPHA as DISKANN_DEFAULT_ALPHA;
use diskann::graph::config::{Builder, MaxDegree};
use diskann::utils::ONE;
use diskann_disk::build::builder::build::DiskIndexBuilder;
use diskann_disk::data_model::AdHoc;
use diskann_disk::data_model::CachingStrategy;
use diskann_disk::disk_index_build_parameter::{DISK_SECTOR_LEN, MemoryBudget, NumPQChunks};
use diskann_disk::search::provider::disk_provider::DiskIndexSearcher;
use diskann_disk::search::provider::disk_vertex_provider_factory::DiskVertexProviderFactory;
use diskann_disk::storage::DiskIndexWriter;
use diskann_disk::storage::disk_index_reader::DiskIndexReader;
use diskann_disk::utils::aligned_file_reader::AlignedFileReaderFactory;
use diskann_disk::{DiskIndexBuildParameters, QuantizationType};
use diskann_providers::model::configuration::IndexConfiguration;
use diskann_providers::storage::{StorageReadProvider, StorageWriteProvider};
use diskann_vector::distance::Metric;

const DISKANN_VERSION: &str = "0.54.0";
// TODO: wire up through config instead of hardcoding (S1-T4)
const NUM_THREADS: usize = 1;
const MAX_POINTS: usize = 1_000_000;
const BUILD_MEMORY_LIMIT_GB: f64 = 2.0;
const BUILD_PQ_CHUNKS: usize = 1;
const MIN_BUILD_POINTS: usize = 1_000;
const SEARCH_IO_LIMIT: usize = 64;

pub struct DiskannIndexFactory {
    diskann_index_path: PathBuf,
    alpha: f32,
}

impl VsIndexFactory for DiskannIndexFactory {
    fn create_index(
        &self,
        index: VsIndexConfiguration,
        table: Arc<RwLock<Table>>,
        memory: mpsc::Sender<Memory>,
    ) -> anyhow::Result<mpsc::Sender<VsIndex>> {
        let params = DiskannParams::try_from((&index, self.alpha, MAX_POINTS))?;
        new(
            params,
            index.key,
            index.dimensions,
            &self.diskann_index_path,
            table,
            memory,
        )
    }

    fn index_engine_version(&self) -> String {
        format!("diskann-{DISKANN_VERSION}")
    }
}

pub fn new_diskann(
    mut config_rx: watch::Receiver<Arc<Config>>,
) -> anyhow::Result<DiskannIndexFactory> {
    let config = config_rx.borrow_and_update();

    let diskann_index_path = config
        .diskann_index_path
        .clone()
        .ok_or_else(|| anyhow::anyhow!("DiskANN index path should be set"))?;

    Ok(DiskannIndexFactory {
        diskann_index_path,
        alpha: config.diskann_alpha.unwrap_or(DISKANN_DEFAULT_ALPHA),
    })
}

fn new(
    params: DiskannParams,
    index_key: IndexKey,
    dimensions: Dimensions,
    diskann_index_path: &Path,
    table: Arc<RwLock<impl TableSearch + Send + Sync + 'static>>,
    _memory: mpsc::Sender<Memory>,
) -> anyhow::Result<mpsc::Sender<VsIndex>> {
    let index_dir = diskann_index_path.join(index_key.as_ref());

    if index_dir.exists() && index_dir.read_dir()?.next().is_some() {
        anyhow::bail!("DiskANN index directory already exists and is non-empty: {index_dir:?}");
    }

    std::fs::create_dir_all(&index_dir).context("failed to create DiskANN index directory")?;

    let collector = InitialBuildCollector::new(&index_dir, dimensions)?;

    let (tx, mut rx) = mpsc::channel(perf::channel_size().into());

    tokio::spawn(perf::hotpath_async(
        {
            async move {
                debug!("starting");
                let mut state = DiskannActorState::Collecting(collector);

                while let Some(msg) = rx.recv().await {
                    state = process_diskann_message(
                        state,
                        msg,
                        &params,
                        dimensions,
                        index_dir.clone(),
                        &table,
                    );
                }

                debug!("finished");
            }
        }
        .instrument(debug_span!("diskann", "{index_key}")),
    ));

    Ok(tx)
}

enum DiskannActorState {
    Collecting(InitialBuildCollector),
    Serving(DiskannServingIndex),
    Failed(String),
}

struct InitialBuildCollector {
    dataset: File,
    dataset_path: PathBuf,
    ids: Vec<(PartitionId, PrimaryId)>,
}

impl InitialBuildCollector {
    fn new(index_dir: &Path, dimensions: Dimensions) -> anyhow::Result<Self> {
        let dataset_path = index_dir.join("dataset.bin");
        let mut dataset = File::create(&dataset_path).map_err(|e| {
            anyhow::anyhow!(
                "failed to create DiskANN dataset at {:?}: {}",
                dataset_path,
                e
            )
        })?;
        let dimensions_u32 = u32::try_from(usize::from(dimensions.0))
            .context("DiskANN dataset dimensions do not fit in u32")?;

        dataset.write_all(&0u32.to_le_bytes())?;
        dataset.write_all(&dimensions_u32.to_le_bytes())?;

        Ok(Self {
            dataset,
            dataset_path,
            ids: Vec::new(),
        })
    }

    fn add(
        &mut self,
        partition_id: PartitionId,
        primary_id: PrimaryId,
        embedding: &Vector,
        dimensions: Dimensions,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            embedding.dim() == Some(dimensions),
            "DiskANN vector dimensions mismatch: expected {}, got {}",
            usize::from(dimensions.0),
            embedding.len()
        );
        self.ids.push((partition_id, primary_id));
        for value in embedding.as_slice() {
            self.dataset.write_all(&value.to_le_bytes())?;
        }
        Ok(())
    }

    fn remove(&mut self, _partition_id: PartitionId, _primary_id: PrimaryId) {
        warn!("DiskANN remove during initial build is not implemented yet");
    }

    fn remove_partition(&mut self, _partition_id: PartitionId) {
        warn!("DiskANN remove partition during initial build is not implemented yet");
    }

    fn finish(mut self) -> anyhow::Result<FinishedDataset> {
        let count = u32::try_from(self.ids.len())
            .context("DiskANN dataset point count does not fit in u32")?;
        self.dataset.seek(SeekFrom::Start(0))?;
        self.dataset.write_all(&count.to_le_bytes())?;
        self.dataset.sync_all()?;

        Ok(FinishedDataset {
            dataset_path: self.dataset_path,
            ids: self.ids,
        })
    }
}

struct FinishedDataset {
    dataset_path: PathBuf,
    ids: Vec<(PartitionId, PrimaryId)>,
}

struct DiskannServingIndex {
    searcher: DiskIndexSearcher<AdHoc<f32, u32>>,
    ids: Vec<(PartitionId, PrimaryId)>,
    space_type: SpaceType,
    l_search_default: NonZeroUsize,
}

fn process_serving_message<T: TableSearch>(
    serving: &mut DiskannServingIndex,
    msg: VsIndex,
    dimensions: Dimensions,
    table: &Arc<RwLock<T>>,
) {
    match msg {
        VsIndex::Ann {
            embedding,
            limit,
            tx,
            ..
        } => {
            let result = ann(serving, embedding, limit, dimensions, table);
            _ = tx.send(result);
        }
        VsIndex::FilteredAnn { tx, .. } => {
            _ = tx.send(Err(anyhow::anyhow!(
                "DiskANN filtered ANN is not implemented yet"
            )));
        }
        VsIndex::Count { tx, .. } => {
            _ = tx.send(Ok(serving.ids.len()));
        }
        VsIndex::AddVector { .. }
        | VsIndex::RemoveVector { .. }
        | VsIndex::RemovePartition { .. } => {
            warn!("DiskANN streaming updates are not implemented yet");
        }
        VsIndex::InitialScanFinished => {}
    }
}

fn ann<T: TableSearch>(
    serving: &DiskannServingIndex,
    embedding: Vector,
    limit: Limit,
    dimensions: Dimensions,
    table: &Arc<RwLock<T>>,
) -> anyhow::Result<(Vec<PrimaryKey>, Vec<Distance>)> {
    anyhow::ensure!(
        embedding.dim() == Some(dimensions),
        "DiskANN query dimensions mismatch: expected {}, got {}",
        usize::from(dimensions.0),
        embedding.len()
    );

    let search = serving
        .searcher
        .search(
            embedding.as_slice(),
            limit.0.get() as u32,
            serving.l_search_default.get() as u32,
            None,
            None,
            false,
        )
        .context("DiskANN search failed")?;

    let table = table.read().unwrap();
    let mut primary_keys = Vec::with_capacity(search.results.len());
    let mut distances = Vec::with_capacity(search.results.len());
    for item in search.results {
        let Some((partition_id, primary_id)) = serving.ids.get(item.vertex_id as usize).copied()
        else {
            continue;
        };
        let Some(primary_key) = table.primary_key(partition_id, primary_id) else {
            continue;
        };
        let distance = Distance::try_from((item.distance, serving.space_type, Some(dimensions)))?;
        primary_keys.push(primary_key);
        distances.push(distance);
    }

    Ok((primary_keys, distances))
}

fn send_failed_response(error: &str, msg: VsIndex) {
    match msg {
        VsIndex::Ann { tx, .. } | VsIndex::FilteredAnn { tx, .. } => {
            _ = tx.send(Err(anyhow::anyhow!(
                "DiskANN index failed to build: {error}"
            )));
        }
        VsIndex::Count { tx, .. } => {
            _ = tx.send(Err(anyhow::anyhow!(
                "DiskANN index failed to build: {error}"
            )));
        }
        VsIndex::AddVector { .. }
        | VsIndex::RemoveVector { .. }
        | VsIndex::RemovePartition { .. }
        | VsIndex::InitialScanFinished => {}
    }
}

fn process_diskann_message<T: TableSearch>(
    state: DiskannActorState,
    msg: VsIndex,
    params: &DiskannParams,
    dimensions: Dimensions,
    index_dir: PathBuf,
    table: &Arc<RwLock<T>>,
) -> DiskannActorState {
    match state {
        DiskannActorState::Collecting(mut collector) => match msg {
            VsIndex::AddVector {
                partition_id,
                primary_id,
                embedding,
                in_progress: _in_progress,
            } => {
                if let Err(err) = collector.add(partition_id, primary_id, &embedding, dimensions) {
                    return DiskannActorState::Failed(err.to_string());
                }
                DiskannActorState::Collecting(collector)
            }
            VsIndex::RemoveVector {
                partition_id,
                primary_id,
                in_progress: _in_progress,
            } => {
                collector.remove(partition_id, primary_id);
                DiskannActorState::Collecting(collector)
            }
            VsIndex::RemovePartition { partition_id } => {
                collector.remove_partition(partition_id);
                DiskannActorState::Collecting(collector)
            }
            VsIndex::InitialScanFinished => match collector.finish() {
                Ok(dataset) => {
                    match build_disk_index(params.clone(), dimensions, index_dir, dataset) {
                        Ok(serving) => DiskannActorState::Serving(serving),
                        Err(err) => DiskannActorState::Failed(err.to_string()),
                    }
                }
                Err(err) => DiskannActorState::Failed(err.to_string()),
            },
            VsIndex::Ann { tx, .. } | VsIndex::FilteredAnn { tx, .. } => {
                _ = tx.send(Err(anyhow::anyhow!(
                    "DiskANN index is still bootstrapping from full scan"
                )));
                DiskannActorState::Collecting(collector)
            }
            VsIndex::Count { tx, .. } => {
                _ = tx.send(Ok(collector.ids.len()));
                DiskannActorState::Collecting(collector)
            }
        },
        DiskannActorState::Serving(mut serving) => {
            process_serving_message(&mut serving, msg, dimensions, table);
            DiskannActorState::Serving(serving)
        }
        DiskannActorState::Failed(error) => {
            send_failed_response(&error, msg);
            DiskannActorState::Failed(error)
        }
    }
}

fn build_disk_index(
    params: DiskannParams,
    dimensions: Dimensions,
    index_dir: PathBuf,
    dataset: FinishedDataset,
) -> anyhow::Result<DiskannServingIndex> {
    anyhow::ensure!(
        dataset.ids.len() >= MIN_BUILD_POINTS,
        "DiskANN requires at least {MIN_BUILD_POINTS} vectors to build; got {}",
        dataset.ids.len()
    );

    let serving = std::thread::spawn(move || {
        build_disk_index_on_current_thread(params, dimensions, index_dir, dataset)
    })
    .join()
    .map_err(|err| anyhow::anyhow!("DiskANN index build thread panicked: {err:?}"))??;

    Ok(serving)
}

fn build_disk_index_on_current_thread(
    params: DiskannParams,
    dimensions: Dimensions,
    index_dir: PathBuf,
    dataset: FinishedDataset,
) -> anyhow::Result<DiskannServingIndex> {
    let storage_provider = NodeLocalSsdProvider::new(index_dir.clone());

    // TODO: make these DiskANN build constants configurable or agree on
    // production defaults when the dedicated DiskANN configuration surface exists.
    let disk_index_build_parameters = DiskIndexBuildParameters::new(
        MemoryBudget::try_from_gb(BUILD_MEMORY_LIMIT_GB)
            .context("failed to create DiskANN build memory budget")?,
        QuantizationType::PQ {
            num_chunks: BUILD_PQ_CHUNKS,
        },
        NumPQChunks::new_with(BUILD_PQ_CHUNKS, usize::from(dimensions.0))
            .context("failed to create DiskANN PQ chunk configuration")?,
    );

    let metric = params.metric;
    let space_type = params.space_type;
    let l_search_default = params.l_search_default;
    let index_configuration = IndexConfiguration::from(params);

    let dataset_path = dataset.dataset_path;
    let prefix_path = index_dir.join("index");

    let dataset_file_str = dataset_path
        .to_str()
        .ok_or_else(|| {
            anyhow::anyhow!("DiskANN dataset path is not valid UTF-8: {dataset_path:?}")
        })?
        .to_string();
    let index_path_prefix_str = prefix_path
        .to_str()
        .ok_or_else(|| {
            anyhow::anyhow!("DiskANN index prefix path is not valid UTF-8: {prefix_path:?}")
        })?
        .to_string();

    let index_writer = DiskIndexWriter::new(
        dataset_file_str,
        index_path_prefix_str,
        None, // No associated data file
        DISK_SECTOR_LEN,
    )
    .context("failed to create a DiskIndexWriter")?;

    let disk_index_pq_pivot_file = index_writer.get_disk_index_pq_pivot_file();
    let disk_index_compressed_pq_file = index_writer.get_disk_index_compressed_pq_file();
    let disk_index_file = index_writer.disk_index_file();

    let mut builder = DiskIndexBuilder::<'_, AdHoc<f32, u32>, _>::new(
        &storage_provider,
        disk_index_build_parameters,
        index_configuration,
        index_writer,
    )
    .map_err(|e| anyhow::anyhow!("failed to create DiskANN index builder: {}", e))?;

    builder.build().context("failed to build DiskANN index")?;

    let disk_index_reader = DiskIndexReader::new(
        disk_index_pq_pivot_file,
        disk_index_compressed_pq_file,
        &storage_provider,
    )
    .context("failed to create DiskANN disk index reader")?;
    let disk_index_path = index_dir.join(disk_index_file);
    let disk_index_path = disk_index_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("DiskANN disk index path is not valid UTF-8"))?
        .to_string();
    let vertex_provider_factory = DiskVertexProviderFactory {
        aligned_reader_factory: AlignedFileReaderFactory::new(disk_index_path),
        caching_strategy: CachingStrategy::None,
        cache: None,
    };
    let searcher = DiskIndexSearcher::new(
        NUM_THREADS,
        SEARCH_IO_LIMIT,
        &disk_index_reader,
        vertex_provider_factory,
        metric,
        None,
    )
    .context("failed to open built DiskANN index")?;

    Ok(DiskannServingIndex {
        searcher,
        ids: dataset.ids,
        space_type,
        l_search_default,
    })
}

#[derive(Clone)]
pub(crate) struct DiskannParams {
    pub(crate) config: DiskannConfig,
    pub(crate) metric: Metric,
    pub(crate) space_type: SpaceType,
    pub(crate) dim: usize,
    pub(crate) max_points: usize,
    #[allow(dead_code)]
    // AFAIK l_search is used per query, but we don't have queries yet, so this is unused for now
    pub(crate) l_search_default: NonZeroUsize,
}

impl TryFrom<(&VsIndexConfiguration, f32, usize)> for DiskannParams {
    type Error = anyhow::Error;

    fn try_from(
        (cfg, alpha, max_points): (&VsIndexConfiguration, f32, usize),
    ) -> Result<Self, Self::Error> {
        anyhow::ensure!(
            cfg.quantization == Quantization::F32,
            "DiskANN engine (v1) only supports F32 quantization; got {:?}",
            cfg.quantization
        );

        let metric: Metric = cfg.space_type.try_into()?;

        let mut builder = Builder::new(
            cfg.connectivity.0,
            MaxDegree::default_slack(),
            cfg.expansion_add.0,
            metric.into(),
        );

        builder.alpha(alpha);

        let config = builder
            .build()
            .context("failed to build DiskANN configuration")?;

        Ok(Self {
            config,
            metric,
            space_type: cfg.space_type,
            dim: usize::from(cfg.dimensions.0),
            max_points,
            l_search_default: NonZeroUsize::new(cfg.expansion_search.0)
                .ok_or_else(|| anyhow::anyhow!("expansion_search must be > 0"))?,
        })
    }
}

impl From<DiskannParams> for IndexConfiguration {
    fn from(params: DiskannParams) -> Self {
        IndexConfiguration::new(
            params.metric,
            params.dim,
            params.max_points,
            ONE,
            NUM_THREADS,
            params.config,
        )
    }
}

impl TryFrom<SpaceType> for Metric {
    type Error = anyhow::Error;

    fn try_from(space_type: SpaceType) -> Result<Self, Self::Error> {
        match space_type {
            SpaceType::Euclidean => Ok(Self::L2),
            SpaceType::Cosine => Ok(Self::Cosine),
            SpaceType::DotProduct => Ok(Self::InnerProduct),
            SpaceType::Hamming => {
                anyhow::bail!("DiskANN does not support Hamming space type")
            }
        }
    }
}

/// A custom storage provider that scopes all DiskANN file operations
/// to a specific directory on the node's local SSD.
pub struct NodeLocalSsdProvider {
    base_dir: PathBuf,
}

impl NodeLocalSsdProvider {
    /// Creates a new provider. The directory should be provisioned before calling this.
    pub fn new(base_dir: PathBuf) -> Self {
        Self { base_dir }
    }

    /// Helper to safely resolve the DiskANN string identifier (e.g., "index.graph")
    /// to a physical path within the SSD directory.
    fn get_path(&self, item_identifier: &str) -> PathBuf {
        self.base_dir.join(item_identifier)
    }
}

impl StorageReadProvider for NodeLocalSsdProvider {
    type Reader = File;

    fn open_reader(&self, item_identifier: &str) -> Result<Self::Reader, std::io::Error> {
        File::open(self.get_path(item_identifier))
    }

    fn get_length(&self, item_identifier: &str) -> Result<u64, std::io::Error> {
        let metadata = std::fs::metadata(self.get_path(item_identifier))?;
        Ok(metadata.len())
    }

    fn exists(&self, item_identifier: &str) -> bool {
        self.get_path(item_identifier).exists()
    }
}

impl StorageWriteProvider for NodeLocalSsdProvider {
    type Writer = File;

    fn open_writer(&self, item_identifier: &str) -> Result<Self::Writer, std::io::Error> {
        OpenOptions::new()
            .write(true)
            .open(self.get_path(item_identifier))
    }

    fn create_for_write(&self, item_identifier: &str) -> Result<Self::Writer, std::io::Error> {
        File::create(self.get_path(item_identifier))
    }

    fn delete(&self, item_identifier: &str) -> Result<(), std::io::Error> {
        let path = self.get_path(item_identifier);
        if path.exists() {
            std::fs::remove_file(path)
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Connectivity;
    use crate::ExpansionAdd;
    use crate::ExpansionSearch;
    use crate::IndexKey;
    use crate::IndexName;
    use crate::KeyspaceName;
    use crate::Quantization;
    use crate::table::MockTableSearch;
    use diskann_providers::storage::{
        get_compressed_pq_file, get_disk_index_file, get_pq_pivot_file,
    };
    use std::num::NonZeroUsize;
    use std::path::Path;
    use tempfile::tempdir;

    const ALPHA: f32 = 1.2;
    const MAX_POINTS: usize = 1_000_000;

    fn test_index() -> VsIndexConfiguration {
        VsIndexConfiguration {
            key: IndexKey::new(
                &KeyspaceName::from("ks".to_string()),
                &IndexName::from("tbl".to_string()),
            ),
            dimensions: NonZeroUsize::new(3).unwrap().into(),
            connectivity: Connectivity(16),
            expansion_add: ExpansionAdd(64),
            expansion_search: ExpansionSearch(32),
            space_type: SpaceType::Euclidean,
            quantization: Quantization::F32,
        }
    }

    #[test]
    fn diskann_metric_try_from_space_type() {
        assert_eq!(Metric::try_from(SpaceType::Euclidean).unwrap(), Metric::L2);
        assert_eq!(Metric::try_from(SpaceType::Cosine).unwrap(), Metric::Cosine);
        assert_eq!(
            Metric::try_from(SpaceType::DotProduct).unwrap(),
            Metric::InnerProduct
        );
        let err = Metric::try_from(SpaceType::Hamming).unwrap_err();
        assert_eq!(
            err.to_string(),
            "DiskANN does not support Hamming space type"
        );
    }

    #[test]
    fn diskann_params_try_from_index_configuration() {
        let params = DiskannParams::try_from((&test_index(), ALPHA, MAX_POINTS)).unwrap();

        assert_eq!(
            params.config.pruned_degree(),
            NonZeroUsize::new(16).unwrap()
        );
        assert_eq!(params.dim, 3);
        assert_eq!(params.l_search_default, NonZeroUsize::new(32).unwrap());
        assert_eq!(params.config.l_build(), NonZeroUsize::new(64).unwrap());
        assert_eq!(params.metric, Metric::L2);
    }

    #[tokio::test]
    async fn new_materializes_disk_provider_files() {
        let tmp_dir = tempdir().unwrap();
        let cfg = test_index();
        let index_key = cfg.key.clone();
        let dimensions = cfg.dimensions;
        let params = DiskannParams::try_from((&cfg, ALPHA, MAX_POINTS)).unwrap();
        let table = Arc::new(RwLock::new(MockTableSearch::new()));
        let (memory_tx, _memory_rx) = mpsc::channel(1);

        let actor = new(
            params,
            index_key.clone(),
            dimensions,
            tmp_dir.path(),
            table,
            memory_tx,
        )
        .unwrap();

        let index_dir = tmp_dir.path().join(index_key.as_ref());
        let index_prefix = index_dir.join("index");
        let index_prefix = index_prefix.to_str().unwrap();

        assert!(index_dir.exists());
        assert!(Path::new(&get_disk_index_file(index_prefix)).exists());
        assert!(Path::new(&get_pq_pivot_file(index_prefix)).exists());
        assert!(Path::new(&get_compressed_pq_file(index_prefix)).exists());

        drop(actor);
    }
}
