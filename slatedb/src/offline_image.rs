//! Offline construction of a new SlateDB database from sorted key/value pairs.
//!
//! This module is intentionally narrow: it only creates a database at an empty
//! path, writes non-overlapping compacted SSTs, and publishes the initial
//! manifest after every SST has been durably uploaded. It is intended for bulk
//! importers that already own ordering, deduplication, and validation.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use bytes::Bytes;
use fail_parallel::FailPointRegistry;
use object_store::{path::Path, ObjectStore};
use slatedb_common::clock::DefaultSystemClock;
use slatedb_common::DbRand;

use crate::block_cache_policy::BlockCachePolicy;
use crate::config::CompressionCodec;
use crate::db_state::{SortedRun, SsTableId, SsTableView};
use crate::filter_policy::FilterPolicy;
use crate::format::sst::SsTableFormat;
use crate::manifest::store::{ManifestStore, StoredManifest};
use crate::manifest::ManifestCore;
use crate::object_store_tag::TableStoreKind;
use crate::tablestore::{EncodedSsTableWriter, TableStore};
use crate::types::{RowEntry, ValueDeletable};
use crate::utils::IdGenerator;
use crate::{Error, PathResolver};

/// Settings for an [`OfflineImageBuilder`].
#[derive(Clone)]
pub struct OfflineImageOptions {
    /// Approximate maximum data-block bytes in each output SST.
    ///
    /// The writer closes an SST after a completed block takes it past this
    /// threshold, so the physical object may be slightly larger.
    pub target_sst_size_bytes: usize,

    /// Block compression for the output SSTs.
    ///
    /// Each SST records its codec, so readers decode the image whatever codec
    /// the database that later opens it is configured with. Set it to the
    /// writer's [`Settings::compression_codec`](crate::config::Settings) so the
    /// image matches the SSTs that compaction writes later.
    pub compression_codec: Option<CompressionCodec>,

    /// Filters built into each output SST.
    ///
    /// Set it to the writer's
    /// [`DbBuilder::with_filter_policies`](crate::DbBuilder::with_filter_policies)
    /// so the image carries the filters that compaction writes later. A reader
    /// drops any stored filter whose policy it is not configured with, so a
    /// policy missing here is one the image's reads can never use.
    pub filter_policies: Vec<Arc<dyn FilterPolicy>>,
}

impl Default for OfflineImageOptions {
    fn default() -> Self {
        Self {
            target_sst_size_bytes: 1024 * 1024 * 1024,
            compression_codec: None,
            filter_policies: SsTableFormat::default().filter_policies,
        }
    }
}

impl std::fmt::Debug for OfflineImageOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OfflineImageOptions")
            .field("target_sst_size_bytes", &self.target_sst_size_bytes)
            .field("compression_codec", &self.compression_codec)
            .field(
                "filter_policies",
                &self
                    .filter_policies
                    .iter()
                    .map(|policy| policy.name())
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

/// Summary returned after an offline database image is published.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OfflineImageResult {
    pub entries: u64,
    pub ssts: usize,
}

/// Streaming builder for a new SlateDB image.
///
/// Keys must be supplied in strictly increasing byte order. The database is
/// invisible until [`finish`](Self::finish) atomically creates its initial
/// manifest. Dropping a builder before `finish` can leave unreferenced SSTs,
/// but can never publish a partial database.
pub struct OfflineImageBuilder {
    manifest_store: Arc<ManifestStore>,
    table_store: Arc<TableStore>,
    clock: Arc<DefaultSystemClock>,
    rand: Arc<DbRand>,
    target_sst_size_bytes: usize,
    current_writer: Option<EncodedSsTableWriter>,
    current_sst_bytes: usize,
    last_key: Option<Bytes>,
    ssts: Vec<SsTableView>,
    entries: u64,
    first_key: Option<Bytes>,
    pending_parts: Arc<AtomicUsize>,
}

/// Build one contiguous key range without publishing a database manifest.
///
/// Callers bound the number and memory of concurrent parts. Return every completed
/// part to its parent in key order. A lost or failed part prevents publication.
pub struct OfflineImagePartBuilder {
    inner: OfflineImageBuilder,
    owner: Arc<AtomicUsize>,
}

/// Completed SSTs that only their original parent can publish.
pub struct OfflineImagePart {
    owner: Arc<AtomicUsize>,
    first_key: Option<Bytes>,
    last_key: Option<Bytes>,
    ssts: Vec<SsTableView>,
    entries: u64,
}

impl OfflineImagePartBuilder {
    /// Add a pair in strictly increasing key order within this part.
    pub async fn add(&mut self, key: Bytes, value: Bytes) -> Result<(), Error> {
        self.inner.add(key, value).await
    }

    /// Finish durable SST uploads without creating a manifest.
    pub async fn finish(mut self) -> Result<OfflineImagePart, Error> {
        self.inner.close_current_sst().await?;
        Ok(OfflineImagePart {
            owner: self.owner,
            first_key: self.inner.first_key,
            last_key: self.inner.last_key,
            ssts: self.inner.ssts,
            entries: self.inner.entries,
        })
    }
}

impl OfflineImageBuilder {
    /// Create a builder at an empty database path.
    pub async fn create(
        path: impl Into<Path>,
        object_store: Arc<dyn ObjectStore>,
        options: OfflineImageOptions,
    ) -> Result<Self, Error> {
        if options.target_sst_size_bytes == 0 {
            return Err(Error::invalid(
                "offline image target_sst_size_bytes must be greater than zero".to_string(),
            ));
        }

        let path = path.into();
        let manifest_store = Arc::new(ManifestStore::new(&path, object_store.clone()));
        let clock = Arc::new(DefaultSystemClock::new());
        if StoredManifest::try_load(manifest_store.clone(), clock.clone())
            .await
            .map_err(Error::from)?
            .is_some()
        {
            return Err(Error::invalid(format!(
                "offline image destination already contains a SlateDB database: {path}"
            )));
        }

        let table_store = Arc::new(TableStore::new_with_fp_registry(
            object_store,
            SsTableFormat {
                compression_codec: options.compression_codec,
                filter_policies: options.filter_policies,
                ..SsTableFormat::default()
            },
            PathResolver::from_root(path),
            Arc::new(FailPointRegistry::new()),
            None,
            TableStoreKind::Main,
            BlockCachePolicy::default(),
        ));

        Ok(Self {
            manifest_store,
            table_store,
            clock,
            rand: Arc::new(DbRand::default()),
            target_sst_size_bytes: options.target_sst_size_bytes,
            current_writer: None,
            current_sst_bytes: 0,
            last_key: None,
            ssts: Vec::new(),
            entries: 0,
            first_key: None,
            pending_parts: Arc::new(AtomicUsize::new(0)),
        })
    }

    /// Create an independent SST writer for a contiguous key range.
    ///
    /// Each part must finish and be accepted before the parent can publish.
    /// Dropping a part does not cancel this requirement.
    pub fn part_builder(&self) -> OfflineImagePartBuilder {
        self.pending_parts.fetch_add(1, Ordering::Relaxed);
        OfflineImagePartBuilder {
            owner: self.pending_parts.clone(),
            inner: Self {
                manifest_store: self.manifest_store.clone(),
                table_store: self.table_store.clone(),
                clock: self.clock.clone(),
                rand: self.rand.clone(),
                target_sst_size_bytes: self.target_sst_size_bytes,
                current_writer: None,
                current_sst_bytes: 0,
                last_key: None,
                ssts: Vec::new(),
                entries: 0,
                first_key: None,
                pending_parts: Arc::new(AtomicUsize::new(0)),
            },
        }
    }

    /// Accept a finished part after all previously accepted keys.
    ///
    /// Foreign, overlapping and out-of-order parts fail. The parent remains
    /// unable to publish after rejecting one of its outstanding parts.
    pub async fn add_part(&mut self, part: OfflineImagePart) -> Result<(), Error> {
        if !Arc::ptr_eq(&self.pending_parts, &part.owner) {
            return Err(Error::invalid(
                "offline image part belongs to another builder".to_string(),
            ));
        }
        if let (Some(last), Some(first)) = (&self.last_key, &part.first_key) {
            if last >= first {
                return Err(Error::invalid(
                    "offline image parts overlap or are out of order".to_string(),
                ));
            }
        }
        let entries = self
            .entries
            .checked_add(part.entries)
            .ok_or_else(|| Error::invalid("offline image entry count overflow".to_string()))?;
        self.close_current_sst().await?;
        if self.first_key.is_none() {
            self.first_key = part.first_key;
        }
        if part.last_key.is_some() {
            self.last_key = part.last_key;
        }
        self.ssts.extend(part.ssts);
        self.entries = entries;
        self.pending_parts.fetch_sub(1, Ordering::Relaxed);
        Ok(())
    }

    /// Add one key/value pair to the image.
    pub async fn add(&mut self, key: Bytes, value: Bytes) -> Result<(), Error> {
        if key.is_empty() {
            return Err(Error::invalid(
                "offline image keys must not be empty".to_string(),
            ));
        }
        validate_entry_lengths(key.len(), value.len())?;
        if self.last_key.as_ref().is_some_and(|last| last >= &key) {
            return Err(Error::invalid(format!(
                "offline image keys must be strictly increasing; previous={:?}, next={:?}",
                self.last_key, key
            )));
        }

        if self.current_writer.is_none() {
            let id = self.rand.rng().gen_ulid(self.clock.as_ref());
            self.current_writer = Some(self.table_store.table_writer(SsTableId::new(id), None));
        }
        let writer = self
            .current_writer
            .as_mut()
            .expect("offline image writer was initialized above");
        if let Some(block_size) = writer
            .add(RowEntry::new(
                key.clone(),
                ValueDeletable::Value(value),
                0,
                None,
                None,
            ))
            .await
            .map_err(Error::from)?
        {
            self.current_sst_bytes += block_size;
        }
        if self.first_key.is_none() {
            self.first_key = Some(key.clone());
        }
        self.last_key = Some(key);
        self.entries += 1;

        if self.current_sst_bytes >= self.target_sst_size_bytes {
            self.close_current_sst().await?;
        }
        Ok(())
    }

    /// Publish the completed image and return its summary.
    pub async fn finish(mut self) -> Result<OfflineImageResult, Error> {
        if self.pending_parts.load(Ordering::Relaxed) != 0 {
            return Err(Error::invalid(
                "offline image has unfinished or unaccepted parts".to_string(),
            ));
        }
        self.close_current_sst().await?;

        let mut core = ManifestCore::new();
        if !self.ssts.is_empty() {
            Arc::make_mut(&mut core.tree)
                .compacted
                .push(SortedRun::new(0, self.ssts.iter().cloned()));
        }
        StoredManifest::create_new_db(self.manifest_store, core, self.clock)
            .await
            .map_err(Error::from)?;

        Ok(OfflineImageResult {
            entries: self.entries,
            ssts: self.ssts.len(),
        })
    }

    async fn close_current_sst(&mut self) -> Result<(), Error> {
        let Some(writer) = self.current_writer.take() else {
            return Ok(());
        };
        let (handle, _) = writer.close().await.map_err(Error::from)?;
        self.ssts.push(SsTableView::identity(handle));
        self.current_sst_bytes = 0;
        Ok(())
    }
}

fn validate_entry_lengths(key_len: usize, value_len: usize) -> Result<(), Error> {
    if key_len > u16::MAX as usize {
        return Err(Error::invalid(format!(
            "offline image key length {key_len} exceeds {} bytes",
            u16::MAX
        )));
    }
    if value_len > u32::MAX as usize {
        return Err(Error::invalid(format!(
            "offline image value length {value_len} exceeds {} bytes",
            u32::MAX
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use bytes::Bytes;
    use object_store::{memory::InMemory, path::Path, ObjectStore};
    use slatedb_common::clock::DefaultSystemClock;

    use super::{validate_entry_lengths, OfflineImageBuilder, OfflineImageOptions};
    use crate::block_cache_policy::BlockCachePolicy;
    use crate::config::{
        CompactionWorkerOptions, CompactorOptions, FlushOptions, FlushType, Settings,
        SizeTieredCompactionSchedulerOptions,
    };
    use crate::filter_policy::{BloomFilterPolicy, FilterPolicy};
    use crate::format::sst::SsTableFormat;
    use crate::manifest::store::{ManifestStore, StoredManifest};
    use crate::object_store_tag::TableStoreKind;
    use crate::prefix_extractor::{PrefixExtractor, PrefixTarget};
    use crate::read_trace::ReadTrace;
    use crate::tablestore::TableStore;
    use crate::{Db, ErrorKind};

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn parallel_parts_publish_one_ordered_database() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("parallel-parts");
        let mut parent = OfflineImageBuilder::create(
            path.clone(),
            store.clone(),
            OfflineImageOptions {
                target_sst_size_bytes: 128,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let tasks = (0..4)
            .map(|part| {
                let mut writer = parent.part_builder();
                tokio::spawn(async move {
                    for index in part * 100..(part + 1) * 100 {
                        writer
                            .add(
                                Bytes::from(format!("key-{index:04}")),
                                Bytes::from(format!("value-{index}")),
                            )
                            .await
                            .unwrap();
                    }
                    writer.finish().await.unwrap()
                })
            })
            .collect::<Vec<_>>();
        for task in tasks {
            parent.add_part(task.await.unwrap()).await.unwrap();
        }
        let result = parent.finish().await.unwrap();
        assert_eq!(result.entries, 400);
        assert!(result.ssts >= 4);
        let db = Db::open(path, store).await.unwrap();
        for index in 0..400 {
            assert_eq!(
                db.get(format!("key-{index:04}").as_bytes()).await.unwrap(),
                Some(Bytes::from(format!("value-{index}")))
            );
        }
        db.close().await.unwrap();
    }

    #[tokio::test]
    async fn incomplete_parts_and_overlapping_parts_cannot_publish() {
        use futures::TryStreamExt;
        for missing in [true, false] {
            let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
            let mut parent =
                OfflineImageBuilder::create("parts", store.clone(), OfflineImageOptions::default())
                    .await
                    .unwrap();
            let mut first = parent.part_builder();
            let mut second = parent.part_builder();
            first
                .add(Bytes::from_static(b"b"), Bytes::from_static(b"1"))
                .await
                .unwrap();
            parent
                .add_part(first.finish().await.unwrap())
                .await
                .unwrap();
            if missing {
                // A dropped writer must remain an outstanding publication obligation.
                drop(second);
            } else {
                second
                    .add(Bytes::from_static(b"b"), Bytes::from_static(b"2"))
                    .await
                    .unwrap();
                assert!(parent
                    .add_part(second.finish().await.unwrap())
                    .await
                    .is_err());
            }
            assert!(parent.finish().await.is_err());
            let objects = store.list(None).try_collect::<Vec<_>>().await.unwrap();
            assert!(objects
                .iter()
                .all(|o| !o.location.as_ref().contains("/manifest/")));
        }
    }

    #[tokio::test]
    async fn foreign_parts_fail_and_empty_parts_work() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let first =
            OfflineImageBuilder::create("first", store.clone(), OfflineImageOptions::default())
                .await
                .unwrap();
        let mut second =
            OfflineImageBuilder::create("second", store, OfflineImageOptions::default())
                .await
                .unwrap();
        assert!(second
            .add_part(first.part_builder().finish().await.unwrap())
            .await
            .is_err());
        assert!(first.finish().await.is_err());
        let empty = second.part_builder().finish().await.unwrap();
        second.add_part(empty).await.unwrap();
        assert_eq!(second.finish().await.unwrap().entries, 0);
    }

    #[tokio::test]
    async fn publishes_a_readable_multi_sst_database() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("offline");
        let mut builder = OfflineImageBuilder::create(
            path.clone(),
            store.clone(),
            OfflineImageOptions {
                target_sst_size_bytes: 1,
                ..OfflineImageOptions::default()
            },
        )
        .await
        .unwrap();
        for index in 0..1_000u32 {
            builder
                .add(
                    Bytes::from(format!("key-{index:04}")),
                    Bytes::from(format!("value-{index}-{}", "x".repeat(64))),
                )
                .await
                .unwrap();
        }
        let result = builder.finish().await.unwrap();
        assert_eq!(result.entries, 1_000);
        assert!(result.ssts > 1);

        let db = Db::open(path, store).await.unwrap();
        assert_eq!(
            db.get(b"key-0042").await.unwrap(),
            Some(Bytes::from(format!("value-42-{}", "x".repeat(64))))
        );
        db.close().await.unwrap();
    }

    #[cfg(feature = "zstd")]
    #[tokio::test]
    async fn compresses_blocks_and_reads_them_back() {
        use futures::TryStreamExt;

        async fn build(
            codec: Option<crate::config::CompressionCodec>,
        ) -> (Arc<dyn ObjectStore>, u64) {
            let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
            let mut builder = OfflineImageBuilder::create(
                Path::from("offline"),
                store.clone(),
                OfflineImageOptions {
                    compression_codec: codec,
                    ..OfflineImageOptions::default()
                },
            )
            .await
            .unwrap();
            for index in 0..10_000u32 {
                builder
                    .add(
                        Bytes::from(format!("cell/c/vprop_idx/name/s:value-{index:08}")),
                        Bytes::from_static(b"1"),
                    )
                    .await
                    .unwrap();
            }
            builder.finish().await.unwrap();
            let bytes = store
                .list(Some(&Path::from("offline/compacted")))
                .map_ok(|meta| meta.size)
                .try_fold(0, |total, size| async move { Ok(total + size) })
                .await
                .unwrap();
            (store, bytes)
        }

        let (_, plain) = build(None).await;
        let (store, compressed) = build(Some(crate::config::CompressionCodec::Zstd)).await;
        assert!(
            compressed * 2 < plain,
            "zstd image is {compressed} bytes, uncompressed {plain}"
        );

        let db = Db::open(Path::from("offline"), store).await.unwrap();
        assert_eq!(
            db.get(b"cell/c/vprop_idx/name/s:value-00004242")
                .await
                .unwrap(),
            Some(Bytes::from_static(b"1"))
        );
        db.close().await.unwrap();
    }

    /// Names of the filters stored in every compacted SST under `root`.
    async fn compacted_filter_names(
        store: Arc<dyn ObjectStore>,
        root: &str,
        policies: Vec<Arc<dyn FilterPolicy>>,
    ) -> Vec<Vec<String>> {
        let manifest = StoredManifest::load(
            Arc::new(ManifestStore::new(&Path::from(root), store.clone())),
            Arc::new(DefaultSystemClock::new()),
        )
        .await
        .unwrap();
        let table_store = TableStore::new(
            store,
            SsTableFormat {
                filter_policies: policies,
                ..SsTableFormat::default()
            },
            Path::from(root),
            None,
            TableStoreKind::Main,
            BlockCachePolicy::default(),
        );
        let mut names = Vec::new();
        for run in &manifest.db_state().tree.compacted {
            for view in run.sst_views() {
                let filters = table_store
                    .read_filters(&view.sst, false, None, &ReadTrace::none(), None)
                    .await
                    .unwrap();
                names.push(filters.iter().map(|filter| filter.name.clone()).collect());
            }
        }
        names
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn writes_the_filters_compaction_writes() {
        struct Fixed3;
        impl PrefixExtractor for Fixed3 {
            fn name(&self) -> &str {
                "fixed3"
            }
            fn prefix_len(&self, target: &PrefixTarget) -> Option<usize> {
                let input = match target {
                    PrefixTarget::Point(key) => key.as_ref(),
                    PrefixTarget::Prefix(prefix) => prefix.as_ref(),
                };
                (input.len() >= 3).then_some(3)
            }
        }
        let policies: Vec<Arc<dyn FilterPolicy>> = vec![
            Arc::new(BloomFilterPolicy::new(10)),
            Arc::new(BloomFilterPolicy::new(10).with_prefix_extractor(Arc::new(Fixed3))),
        ];
        // Above the writer's default `min_filter_keys`, so compaction output
        // carries filters at all.
        let keys = (0..4_000u32)
            .map(|index| Bytes::from(format!("key-{index:05}")))
            .collect::<Vec<_>>();
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());

        let mut builder = OfflineImageBuilder::create(
            Path::from("image"),
            store.clone(),
            OfflineImageOptions {
                filter_policies: policies.clone(),
                ..OfflineImageOptions::default()
            },
        )
        .await
        .unwrap();
        for key in &keys {
            builder
                .add(key.clone(), Bytes::from_static(b"v"))
                .await
                .unwrap();
        }
        builder.finish().await.unwrap();

        // Parts use the same filter policies as the parent builder.
        let mut parts = OfflineImageBuilder::create(
            Path::from("parts"),
            store.clone(),
            OfflineImageOptions {
                filter_policies: policies.clone(),
                ..OfflineImageOptions::default()
            },
        )
        .await
        .unwrap();
        for keys in keys.chunks(2_000) {
            let mut part = parts.part_builder();
            for key in keys {
                part.add(key.clone(), Bytes::from_static(b"v"))
                    .await
                    .unwrap();
            }
            parts.add_part(part.finish().await.unwrap()).await.unwrap();
        }
        parts.finish().await.unwrap();

        let settings = Settings {
            flush_interval: Some(Duration::from_millis(100)),
            manifest_poll_interval: Duration::from_millis(100),
            compactor_options: Some(CompactorOptions {
                poll_interval: Duration::from_millis(100),
                scheduler_options: SizeTieredCompactionSchedulerOptions {
                    min_compaction_sources: 1,
                    ..SizeTieredCompactionSchedulerOptions::default()
                }
                .into(),
                worker: Some(CompactionWorkerOptions {
                    compactions_poll_interval: Duration::from_millis(100),
                    ..CompactionWorkerOptions::default()
                }),
                ..CompactorOptions::default()
            }),
            ..Settings::default()
        };
        let db = Db::builder("loaded", store.clone())
            .with_settings(settings)
            .with_filter_policies(policies.clone())
            .build()
            .await
            .unwrap();
        for key in &keys {
            db.put(key, b"v").await.unwrap();
        }
        db.flush_with_options(FlushOptions {
            flush_type: FlushType::MemTable,
        })
        .await
        .unwrap();
        let manifest_store = Arc::new(ManifestStore::new(&Path::from("loaded"), store.clone()));
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        loop {
            let manifest =
                StoredManifest::load(manifest_store.clone(), Arc::new(DefaultSystemClock::new()))
                    .await
                    .unwrap();
            let tree = &manifest.db_state().tree;
            if tree.l0.is_empty() && !tree.compacted.is_empty() {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "the L0 was not compacted"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        db.close().await.unwrap();

        let image = compacted_filter_names(store.clone(), "image", policies.clone()).await;
        let parts = compacted_filter_names(store.clone(), "parts", policies.clone()).await;
        let compacted = compacted_filter_names(store, "loaded", policies).await;
        assert_eq!(parts, vec![image[0].clone(), image[0].clone()]);
        assert_eq!(
            image,
            vec![vec!["_bf".to_string(), "_bf:p=fixed3".to_string()]]
        );
        assert_eq!(image, compacted);
    }

    #[tokio::test]
    async fn rejects_unsorted_keys_and_existing_database() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("offline");
        let mut builder = OfflineImageBuilder::create(
            path.clone(),
            store.clone(),
            OfflineImageOptions::default(),
        )
        .await
        .unwrap();
        builder
            .add(Bytes::from_static(b"b"), Bytes::from_static(b"1"))
            .await
            .unwrap();
        let err = builder
            .add(Bytes::from_static(b"a"), Bytes::from_static(b"2"))
            .await
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Invalid);
        builder.finish().await.unwrap();

        let err = OfflineImageBuilder::create(path, store, OfflineImageOptions::default())
            .await
            .err()
            .expect("existing database must be rejected");
        assert_eq!(err.kind(), ErrorKind::Invalid);
    }

    #[tokio::test]
    async fn rejects_encoder_lengths_as_invalid_input() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let mut builder = OfflineImageBuilder::create(
            Path::from("oversized"),
            store,
            OfflineImageOptions::default(),
        )
        .await
        .unwrap();
        let error = builder
            .add(Bytes::from(vec![b'k'; u16::MAX as usize + 1]), Bytes::new())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Invalid);
        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            validate_entry_lengths(1, u32::MAX as usize + 1)
                .unwrap_err()
                .kind(),
            ErrorKind::Invalid
        );
    }
}
