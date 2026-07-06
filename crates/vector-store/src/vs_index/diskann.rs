/*
 * Copyright 2025-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.0
 */

use crate::Config;
use crate::SpaceType;
use crate::VsIndexFactory;
use crate::indexes::NeedsFiltering::No;
use crate::memory::Memory;
use crate::perf;
use crate::table::Table;
use crate::vs_index::actor::CountR;
use crate::vs_index::actor::VsIndex;
use crate::vs_index::factory::VsIndexConfiguration;
use anyhow::Context;
use anyhow::anyhow;
use diskann_providers::index::diskann_async;
use diskann_providers::model::graph::provider::async_::inmem::SetStartPoints;
use diskann_providers::storage::AsyncIndexMetadata;
use diskann_providers::storage::FileStorageProvider;
use diskann_providers::storage::SaveWith;
use diskann_vector::distance::Metric;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::RwLock;
use tokio::runtime::Handle;
use tokio::sync::mpsc;
use tokio::sync::watch;
use tokio::task;
use tracing::Instrument;
use tracing::debug;
use tracing::debug_span;
use tracing::warn;

const DISKANN_VERSION: &str = "v0.54.0";
const DISKANN_MAX_POINTS: usize = 1_000_000;
const DISKANN_DEFAULT_ALPHA: f64 = 1.2;

pub struct DiskannIndexFactory {
    index_path: PathBuf,
    alpha: f64,
}

struct DiskannParameters {
    pruned_degree: usize,
    l_build: usize,
    beam_width: usize,
    metric: Metric,
}

impl VsIndexFactory for DiskannIndexFactory {
    fn create_index(
        &self,
        index: VsIndexConfiguration,
        _table: Arc<RwLock<Table>>,
        _memory: mpsc::Sender<Memory>,
    ) -> anyhow::Result<mpsc::Sender<VsIndex>> {
        new(index, &self.index_path, self.alpha)
    }

    fn index_engine_version(&self) -> String {
        format!("diskann-{DISKANN_VERSION}")
    }
}

pub fn new_diskann(
    mut config_rx: watch::Receiver<Arc<Config>>,
) -> anyhow::Result<DiskannIndexFactory> {
    let _config = config_rx.borrow_and_update().clone();
    // TODO: Wire through Config
    let index_path: PathBuf = "/var/tmp/vector-store/diskann".into();
    Ok(DiskannIndexFactory {
        index_path,
        alpha: DISKANN_DEFAULT_ALPHA,
    })
}

fn new(
    index: VsIndexConfiguration,
    base_index_path: &Path,
    alpha: f64,
) -> anyhow::Result<mpsc::Sender<VsIndex>> {
    let params = diskann_parameters(&index)?;
    let index_dir = base_index_path.join(&index.key.as_ref());
    fs::create_dir_all(&index_dir).with_context(|| {
        format!(
            "failed to create DiskANN index directory {}",
            index_dir.display()
        )
    })?;

    let mut builder = diskann::graph::config::Builder::new_with(
        params.pruned_degree,
        diskann::graph::config::MaxDegree::default_slack(),
        params.l_build,
        params.metric.into(),
        |_| {},
    );
    builder.alpha(alpha as f32);

    let config = builder
        .build()
        .context("failed to build DiskANN configuration")?;

    let provider_parameters =
        diskann_providers::model::graph::provider::async_::inmem::DefaultProviderParameters {
            max_points: DISKANN_MAX_POINTS,
            frozen_points: diskann::utils::ONE,
            dim: index.dimensions.0.get(),
            metric: params.metric,
            prefetch_lookahead: None,
            prefetch_cache_line_level: None,
            max_degree: config.max_degree_u32().get(),
        };

    let diskann_index = diskann_async::new_index::<f32, _>(
        config,
        provider_parameters,
        diskann_providers::model::graph::provider::async_::common::NoDeletes,
    )
    .context("failed to create DiskANN index")?;

    // let start_point = vec![0.0f32; index.dimensions.0.get()];
    // diskann_index
    //     .provider()
    //     .set_start_points(std::iter::once(start_point.as_slice()))
    //     .context("failed to initialize DiskANN start point")?;

    // persist_index(&diskann_index, &index_dir)?;

    let index_key = index.key.clone();
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
                            _ = tx.send(Ok((vec![], vec![])));
                        }
                        VsIndex::Count { tx, .. } => {
                            _ = tx.send(Ok(0));
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

// fn persist_index(
//     index: &diskann_providers::index::diskann_async::MemoryIndex<f32>,
//     index_dir: &Path,
// ) -> anyhow::Result<()> {
//     let storage = FileStorageProvider;
//     let prefix = index_dir.join("index");
//     let metadata = AsyncIndexMetadata::new(path_to_string(&prefix)?);
//     task::block_in_place(|| {
//         Handle::current().block_on(async { index.save_with(&storage, &metadata).await })
//     })
//     .context("failed to persist DiskANN index to disk")
// }

// fn path_to_string(path: &Path) -> anyhow::Result<String> {
//     path.to_str()
//         .map(str::to_owned)
//         .ok_or_else(|| anyhow!("path is not valid UTF-8: {}", path.display()))
// }

fn diskann_parameters(index: &VsIndexConfiguration) -> anyhow::Result<DiskannParameters> {
    Ok(DiskannParameters {
        pruned_degree: index.connectivity.0,
        l_build: index.expansion_add.0,
        beam_width: index.expansion_search.0,
        metric: diskann_metric(index.space_type)?,
    })
}

fn diskann_metric(space_type: SpaceType) -> anyhow::Result<Metric> {
    match space_type {
        SpaceType::Euclidean => Ok(Metric::L2),
        SpaceType::Cosine => Ok(Metric::Cosine),
        SpaceType::DotProduct => Ok(Metric::InnerProduct),
        SpaceType::Hamming => anyhow::bail!("DiskANN does not support Hamming space type"),
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
    use tempfile::TempDir;

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
    fn diskann_metric_maps_supported_space_types() {
        assert_eq!(diskann_metric(SpaceType::Euclidean).unwrap(), Metric::L2);
        assert_eq!(diskann_metric(SpaceType::Cosine).unwrap(), Metric::Cosine);
        assert_eq!(
            diskann_metric(SpaceType::DotProduct).unwrap(),
            Metric::InnerProduct
        );
        assert!(diskann_metric(SpaceType::Hamming).is_err());
    }

    #[test]
    fn diskann_parameters_follow_sep_mapping() {
        let params = diskann_parameters(&test_index()).unwrap();
        assert_eq!(params.pruned_degree, 16);
        assert_eq!(params.l_build, 64);
        assert_eq!(params.beam_width, 32);
        assert_eq!(params.metric, Metric::L2);
    }

    #[tokio::test]
    async fn new_diskann_requires_index_path() {
        let (_, rx) = watch::channel(Arc::new(Config::default()));
        let err = new_diskann(rx).err().unwrap();
        assert!(
            err.to_string()
                .contains("VECTOR_STORE_DISKANN_INDEX_PATH must be set")
        );
    }

    #[tokio::test]
    async fn create_index_materializes_diskann_files() {
        let temp_dir = TempDir::new().unwrap();
        let factory = DiskannIndexFactory {
            index_path: temp_dir.path().to_path_buf(),
            alpha: DISKANN_DEFAULT_ALPHA,
        };
        let (memory_tx, _memory_rx) = mpsc::channel(1);
        let index = test_index();
        let actor = factory
            .create_index(
                test_index(),
                Arc::new(RwLock::new(test_table(index.key.clone()))),
                memory_tx,
            )
            .unwrap();
        drop(actor);

        let index_dir = temp_dir.path().join(index.key.as_ref());
        let prefix = index_dir.join("index");
        assert!(index_dir.exists());
        assert!(prefix.exists());
        assert!(prefix.with_extension("data").exists());
    }

    // #[tokio::test]
    // async fn load_config_parses_diskann_settings() {
    //     let temp_dir = TempDir::new().unwrap();
    //     let env = |key: &str| match key {
    //         "VECTOR_STORE_DISKANN_INDEX_PATH" => Ok(temp_dir.path().display().to_string()),
    //         "VECTOR_STORE_DISKANN_ALPHA" => Ok("1.5".to_string()),
    //         _ => Err(anyhow!("env var {key} not found")),
    //     };
    //     let config = crate::config_manager::load_config(env).await.unwrap();
    //     assert_eq!(
    //         config.diskann_index_path,
    //         Some(temp_dir.path().to_path_buf())
    //     );
    //     assert_eq!(config.diskann_alpha, Some(1.5));
    // }

    fn test_table(index_key: IndexKey) -> Table {
        use crate::ColumnName;
        use crate::NonemptyArc;
        use scylla::cluster::metadata::NativeType;
        use std::collections::HashMap;

        Table::new(
            index_key,
            NonemptyArc::new([ColumnName::from("pk")]).unwrap(),
            1,
            None,
            &[],
            Arc::new(HashMap::<ColumnName, NativeType>::new()),
        )
        .unwrap()
    }
}
