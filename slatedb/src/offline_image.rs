//! Offline construction of a new SlateDB database from sorted key/value pairs.
//!
//! This module is intentionally narrow: it only creates a database at an empty
//! path, writes non-overlapping compacted SSTs, and publishes the initial
//! manifest after every SST has been durably uploaded. It is intended for bulk
//! importers that already own ordering, deduplication, and validation.

use std::sync::Arc;

use bytes::Bytes;
use fail_parallel::FailPointRegistry;
use object_store::{path::Path, ObjectStore};
use slatedb_common::clock::DefaultSystemClock;
use ulid::Ulid;

use crate::block_cache_policy::BlockCachePolicy;
use crate::db_state::{SortedRun, SsTableId, SsTableView};
use crate::format::sst::SsTableFormat;
use crate::manifest::store::{ManifestStore, StoredManifest};
use crate::manifest::ManifestCore;
use crate::object_store_tag::TableStoreKind;
use crate::object_stores::ObjectStores;
use crate::tablestore::{EncodedSsTableWriter, TableStore};
use crate::types::{RowEntry, ValueDeletable};
use crate::{Error, PathResolver};

/// Settings for an [`OfflineImageBuilder`].
#[derive(Clone, Debug)]
pub struct OfflineImageOptions {
    /// Approximate maximum data-block bytes in each output SST.
    ///
    /// The writer closes an SST after a completed block takes it past this
    /// threshold, so the physical object may be slightly larger.
    pub target_sst_size_bytes: usize,
}

impl Default for OfflineImageOptions {
    fn default() -> Self {
        Self {
            target_sst_size_bytes: 1024 * 1024 * 1024,
        }
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
    target_sst_size_bytes: usize,
    current_writer: Option<EncodedSsTableWriter>,
    current_sst_bytes: usize,
    last_key: Option<Bytes>,
    ssts: Vec<SsTableView>,
    entries: u64,
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
        if StoredManifest::try_load(manifest_store.clone(), clock)
            .await
            .map_err(Error::from)?
            .is_some()
        {
            return Err(Error::invalid(format!(
                "offline image destination already contains a SlateDB database: {path}"
            )));
        }

        let table_store = Arc::new(TableStore::new_with_fp_registry(
            ObjectStores::new(object_store, None),
            SsTableFormat::default(),
            PathResolver::from_root(path),
            Arc::new(FailPointRegistry::new()),
            None,
            TableStoreKind::Main,
            BlockCachePolicy::default(),
        ));

        Ok(Self {
            manifest_store,
            table_store,
            target_sst_size_bytes: options.target_sst_size_bytes,
            current_writer: None,
            current_sst_bytes: 0,
            last_key: None,
            ssts: Vec::new(),
            entries: 0,
        })
    }

    /// Add one key/value pair to the image.
    pub async fn add(&mut self, key: Bytes, value: Bytes) -> Result<(), Error> {
        if key.is_empty() {
            return Err(Error::invalid(
                "offline image keys must not be empty".to_string(),
            ));
        }
        if self.last_key.as_ref().is_some_and(|last| last >= &key) {
            return Err(Error::invalid(format!(
                "offline image keys must be strictly increasing; previous={:?}, next={:?}",
                self.last_key, key
            )));
        }

        let writer = self.current_writer.get_or_insert_with(|| {
            self.table_store
                .table_writer(SsTableId::Compacted(Ulid::new()))
        });
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
        self.last_key = Some(key);
        self.entries += 1;

        if self.current_sst_bytes >= self.target_sst_size_bytes {
            self.close_current_sst().await?;
        }
        Ok(())
    }

    /// Publish the completed image and return its summary.
    pub async fn finish(mut self) -> Result<OfflineImageResult, Error> {
        self.close_current_sst().await?;

        let mut core = ManifestCore::new();
        if !self.ssts.is_empty() {
            Arc::make_mut(&mut core.tree)
                .compacted
                .push(SortedRun::new(0, self.ssts.iter().cloned()));
        }
        StoredManifest::create_new_db(
            self.manifest_store,
            core,
            Arc::new(DefaultSystemClock::new()),
        )
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
        let handle = writer.close().await.map_err(Error::from)?;
        self.ssts.push(SsTableView::identity(handle));
        self.current_sst_bytes = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bytes::Bytes;
    use object_store::{memory::InMemory, path::Path, ObjectStore};

    use super::{OfflineImageBuilder, OfflineImageOptions};
    use crate::{Db, ErrorKind};

    #[tokio::test]
    async fn publishes_a_readable_multi_sst_database() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("offline");
        let mut builder = OfflineImageBuilder::create(
            path.clone(),
            store.clone(),
            OfflineImageOptions {
                target_sst_size_bytes: 1,
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
}
