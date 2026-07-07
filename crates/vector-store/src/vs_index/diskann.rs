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

