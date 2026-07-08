/*
 * Copyright 2025-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.0
 */

use crate::Config;
use crate::Quantization;
use crate::SpaceType;
use crate::VsIndexFactory;
use crate::memory::Memory;
use crate::perf;
use crate::table::Table;
use crate::vs_index::actor::VsIndex;
use crate::vs_index::factory::VsIndexConfiguration;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::RwLock;
use tokio::sync::mpsc;
use tokio::sync::watch;
use tracing::Instrument;
use tracing::debug;
use tracing::debug_span;
use tracing::warn;
use diskann::utils::ONE;
use diskann_providers::model::configuration::IndexConfiguration;
use diskann_vector::distance::Metric;

const DISKANN_VERSION: &str = "0.54.0";
const NUM_THREADS: usize = 4;

pub struct DiskannIndexFactory;

impl VsIndexFactory for DiskannIndexFactory {
    fn create_index(
        &self,
        index: VsIndexConfiguration,
        _table: Arc<RwLock<Table>>,
        _memory: mpsc::Sender<Memory>,
    ) -> anyhow::Result<mpsc::Sender<VsIndex>> {
        new(index.key)
    }

    fn index_engine_version(&self) -> String {
        format!("diskann-{DISKANN_VERSION}")
    }
}

pub fn new_diskann(
    _config_rx: watch::Receiver<Arc<Config>>,
) -> anyhow::Result<DiskannIndexFactory> {
    Ok(DiskannIndexFactory)
}

fn new(index_key: crate::IndexKey) -> anyhow::Result<mpsc::Sender<VsIndex>> {
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

pub(crate) struct DiskannParams {
    pub(crate) config: DiskannConfig,
    pub(crate) metric: Metric,
    pub(crate) dim: usize,
    pub(crate) max_points: usize,
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
    use std::num::NonZeroUsize;

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
        assert!(Metric::try_from(SpaceType::Hamming).is_err());
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
}
