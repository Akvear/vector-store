/*
 * Copyright 2025-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.0
 */

use crate::Config;
use crate::Dimensions;
use crate::IndexKey;
use crate::Quantization;
use crate::SpaceType;
use crate::VsIndexFactory;
use crate::memory::Memory;
use crate::perf;
use crate::table::Table;
use crate::table::TableSearch;
use crate::vs_index::actor::VsIndex;
use crate::vs_index::factory::VsIndexConfiguration;

use std::fs::{File, OpenOptions};
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
use diskann_disk::disk_index_build_parameter::{DISK_SECTOR_LEN, MemoryBudget, NumPQChunks};
use diskann_disk::storage::DiskIndexWriter;
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
const SEED_DATASET_POINTS: u32 = 256;

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
    _table: Arc<RwLock<impl TableSearch + Send + Sync + 'static>>,
    _memory: mpsc::Sender<Memory>,
) -> anyhow::Result<mpsc::Sender<VsIndex>> {
    let index_dir = diskann_index_path.join(index_key.as_ref());

    if index_dir.exists() && index_dir.read_dir()?.next().is_some() {
        anyhow::bail!("DiskANN index directory already exists and is non-empty: {index_dir:?}");
    }

    std::fs::create_dir_all(&index_dir).context("failed to create DiskANN index directory")?;

    build_seed_disk_index(params, dimensions, index_dir.clone())?;

    let (tx, mut rx) = mpsc::channel(perf::channel_size().into());

    tokio::spawn(perf::hotpath_async(
        {
            let _index_key = index_key.clone();
            async move {
                debug!("starting");

                while let Some(msg) = rx.recv().await {
                    match msg {
                        VsIndex::AddVector { .. }
                        | VsIndex::RemoveVector { .. }
                        | VsIndex::RemovePartition { .. } => {
                            warn!("not implemented yet");
                        }
                        VsIndex::Ann { tx, .. } | VsIndex::FilteredAnn { tx, .. } => {
                            _ = tx
                                .send(Err(anyhow::anyhow!("DiskANN index is not implemented yet")));
                        }
                        VsIndex::Count { tx, .. } => {
                            _ = tx
                                .send(Err(anyhow::anyhow!("DiskANN index is not implemented yet")));
                        }
                    }
                }

                debug!("finished");
            }
        }
        .instrument(debug_span!("diskann", "{index_key}")),
    ));

    Ok(tx)
}

fn build_seed_disk_index(
    params: DiskannParams,
    dimensions: Dimensions,
    index_dir: PathBuf,
) -> anyhow::Result<()> {
    std::thread::spawn(move || {
        build_seed_disk_index_on_current_thread(params, dimensions, index_dir)
    })
    .join()
    .map_err(|err| anyhow::anyhow!("DiskANN seed index build thread panicked: {err:?}"))?
}

fn build_seed_disk_index_on_current_thread(
    params: DiskannParams,
    dimensions: Dimensions,
    index_dir: PathBuf,
) -> anyhow::Result<()> {
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

    let index_configuration = IndexConfiguration::from(params);

    let dataset_path = index_dir.join("dummy_dataset.bin");
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

    write_seed_dataset(&dataset_path, dimensions)?;

    let index_writer = DiskIndexWriter::new(
        dataset_file_str,
        index_path_prefix_str,
        None, // No associated data file
        DISK_SECTOR_LEN,
    )
    .context("failed to create a DiskIndexWriter")?;

    let mut builder = DiskIndexBuilder::<'_, AdHoc<f32, u32>, _>::new(
        &storage_provider,
        disk_index_build_parameters,
        index_configuration,
        index_writer,
    )
    .map_err(|e| anyhow::anyhow!("failed to create DiskANN index builder: {}", e))?;

    builder.build().context("failed to build DiskANN index")?;

    Ok(())
}

fn write_seed_dataset(dataset_path: &Path, dimensions: Dimensions) -> anyhow::Result<()> {
    // TODO: Important
    // DiskANN3's disk builder trains PQ during construction and cannot build from
    // an empty dataset. Vector Store creates the index before CDC supplies real
    // vectors, so S1-T3 materializes disk-provider files using deterministic seed
    // vectors. This must be discussed and replaced with an agreed approach.
    let mut dataset = File::create(dataset_path).map_err(|e| {
        anyhow::anyhow!(
            "failed to create DiskANN seed dataset at {:?}: {}",
            dataset_path,
            e
        )
    })?;
    let dimensions_u32 = u32::try_from(usize::from(dimensions.0))
        .context("DiskANN seed dataset dimensions do not fit in u32")?;

    dataset.write_all(&SEED_DATASET_POINTS.to_le_bytes())?;
    dataset.write_all(&dimensions_u32.to_le_bytes())?;

    for i in 0..SEED_DATASET_POINTS {
        for j in 0..usize::from(dimensions.0) {
            let dummy_val: f32 = (i as f32) + (j as f32 * 0.1);
            dataset.write_all(&dummy_val.to_le_bytes())?;
        }
    }

    dataset.sync_all()?;
    Ok(())
}

pub(crate) struct DiskannParams {
    pub(crate) config: DiskannConfig,
    pub(crate) metric: Metric,
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
