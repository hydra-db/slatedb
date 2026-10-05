use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::TryStreamExt;
use object_store::memory::InMemory;
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetRange, GetResult, ListResult, MultipartUpload, ObjectMeta,
    ObjectStore, ObjectStoreExt, PutMultipartOptions, PutOptions, PutPayload, PutResult,
};
use tokio::sync::{mpsc, Semaphore};

use super::{SlateDbWalIterator, SlateDbWalIteratorOptions, WalIteratorEndBound};
use crate::format::sst::SsTableFormat;
use crate::manifest::{Manifest, ManifestCore, VersionedManifest};
use crate::object_store_tag::TableStoreKind;
use crate::types::RowEntry;
use crate::wal::slatedb::sst_iterator::WalSstIteratorOptions;
use crate::wal::slatedb::store::WalTableStore;
use crate::wal::{WalError, WalIterator as _};

struct FutureManifest;

#[async_trait]
impl super::ManifestReader for FutureManifest {
    async fn manifest(&self) -> Result<VersionedManifest, crate::error::SlateDBError> {
        let mut core = ManifestCore::new();
        core.next_wal_sst_id = 1;
        Ok(VersionedManifest::from_manifest(1, Manifest::initial(core)))
    }
}

#[derive(Debug)]
struct PayloadGateStore {
    inner: InMemory,
    gates: BTreeMap<u64, Semaphore>,
    started: mpsc::UnboundedSender<u64>,
    active: AtomicUsize,
    peak: AtomicUsize,
}

struct ActivePayload<'a>(&'a AtomicUsize);
impl Drop for ActivePayload<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

impl fmt::Display for PayloadGateStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "payload-gates")
    }
}

#[async_trait]
impl ObjectStore for PayloadGateStore {
    async fn get_opts(&self, key: &Path, opts: GetOptions) -> object_store::Result<GetResult> {
        if matches!(&opts.range, Some(GetRange::Bounded(range)) if range.start == 0) {
            let id = key
                .filename()
                .unwrap()
                .strip_suffix(".sst")
                .unwrap()
                .parse::<u64>()
                .unwrap();
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(active, Ordering::SeqCst);
            let _active = ActivePayload(&self.active);
            let _ = self.started.send(id);
            self.gates[&id].acquire().await.unwrap().forget();
        }
        self.inner.get_opts(key, opts).await
    }
    async fn put_opts(
        &self,
        key: &Path,
        data: PutPayload,
        opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        self.inner.put_opts(key, data, opts).await
    }
    async fn put_multipart_opts(
        &self,
        key: &Path,
        opts: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(key, opts).await
    }
    fn delete_stream(
        &self,
        keys: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        self.inner.delete_stream(keys)
    }
    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.inner.list(prefix)
    }
    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> object_store::Result<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }
    async fn copy_opts(
        &self,
        from: &Path,
        to: &Path,
        opts: CopyOptions,
    ) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, opts).await
    }
}

async fn gated_fixture() -> (
    Arc<PayloadGateStore>,
    Arc<WalTableStore>,
    mpsc::UnboundedReceiver<u64>,
) {
    let (started, rx) = mpsc::unbounded_channel();
    let object_store = Arc::new(PayloadGateStore {
        inner: InMemory::new(),
        gates: (1..=4).map(|id| (id, Semaphore::new(0))).collect(),
        started,
        active: AtomicUsize::new(0),
        peak: AtomicUsize::new(0),
    });
    let store = Arc::new(WalTableStore::new(
        object_store.clone(),
        SsTableFormat::default(),
        Path::from("wal-prefetch"),
        TableStoreKind::Main,
    ));
    for id in 1..=4 {
        let mut builder = store.table_builder();
        builder
            .add(RowEntry::new_value(
                format!("key-{id}").as_bytes(),
                b"value",
                id,
            ))
            .await
            .unwrap();
        let encoded = builder.build().await.unwrap();
        store.write_sst(id, &encoded).await.unwrap();
    }
    (object_store, store, rx)
}

fn options() -> SlateDbWalIteratorOptions {
    SlateDbWalIteratorOptions {
        sst_batch_size: 3,
        sst_iter_options: WalSstIteratorOptions {
            target_bytes_to_fetch: 1024 * 1024,
        },
    }
}

#[tokio::test(start_paused = true)]
async fn future_payloads_prefetch_concurrently_but_cancelled_next_keeps_order() {
    let (gates, store, mut started) = gated_fixture().await;
    let mut iter =
        SlateDbWalIterator::range(1, WalIteratorEndBound::Exclusive(5), options(), store).unwrap();
    let mut first = Box::pin(iter.next());
    let mut observed = BTreeSet::new();
    tokio::time::timeout(Duration::from_secs(1), async {
        while observed.len() != 3 {
            tokio::select! {
                row = &mut first => panic!("first WAL returned before payload release: {}", row.is_ok()),
                id = started.recv() => { observed.insert(id.unwrap()); }
            }
        }
    }).await.expect("later WAL payloads were not prefetched while the first was blocked");
    assert_eq!(observed, BTreeSet::from([1, 2, 3]));
    assert_eq!(gates.peak.load(Ordering::SeqCst), 3);
    // Complete later files first. Cancelling next must retain the first task
    // and its position, rather than returning or losing a later WAL.
    gates.gates[&2].add_permits(1);
    gates.gates[&3].add_permits(1);
    assert!(tokio::time::timeout(Duration::from_millis(10), &mut first)
        .await
        .is_err());
    drop(first);
    gates.gates[&1].add_permits(1);
    gates.gates[&4].add_permits(1);
    for id in 1..=4 {
        let batch = iter.next().await.unwrap().unwrap();
        assert_eq!(batch.last_consumed_wal_file_id, id);
        assert_eq!(batch.rows.len(), 1);
        assert_eq!(batch.rows[0].seq, id);
    }
    assert!(iter.next().await.unwrap().is_none());
    assert_eq!(gates.peak.load(Ordering::SeqCst), 3);
}

#[tokio::test(start_paused = true)]
async fn dropping_iterator_aborts_pending_payload_prefetches() {
    let (gates, store, mut started) = gated_fixture().await;
    let mut iter =
        SlateDbWalIterator::range(1, WalIteratorEndBound::Exclusive(5), options(), store).unwrap();
    let mut first = Box::pin(iter.next());
    for _ in 0..3 {
        tokio::select! {
            _ = &mut first => panic!("WAL returned while every payload was gated"),
            id = started.recv() => { assert!(id.is_some()); }
        }
    }
    drop(first);
    drop(iter);
    for _ in 0..20 {
        tokio::task::yield_now().await;
    }
    assert_eq!(gates.active.load(Ordering::SeqCst), 0);
    assert!(started.try_recv().is_err());
}

#[tokio::test]
async fn a_later_prefetch_error_does_not_hide_earlier_wals() {
    let store = Arc::new(WalTableStore::new(
        Arc::new(InMemory::new()),
        SsTableFormat::default(),
        Path::from("wal-prefetch-errors"),
        TableStoreKind::Main,
    ));
    let mut builder = store.table_builder();
    builder
        .add(RowEntry::new_value(b"key", b"value", 1))
        .await
        .unwrap();
    let encoded = builder.build().await.unwrap();
    store.write_sst(1, &encoded).await.unwrap();
    store.write_wal_fence(3).await.unwrap();
    let mut iter =
        SlateDbWalIterator::range(1, WalIteratorEndBound::Exclusive(4), options(), store).unwrap();
    assert_eq!(
        iter.next()
            .await
            .unwrap()
            .unwrap()
            .last_consumed_wal_file_id,
        1
    );
    assert!(matches!(iter.next().await, Err(WalError::WalTruncated(2))));
    assert!(matches!(iter.next().await, Err(WalError::WalTruncated(2))));
}

#[tokio::test(start_paused = true)]
async fn an_existing_wal_lost_during_payload_prefetch_is_truncation_not_future_polling() {
    let (gates, store, mut started) = gated_fixture().await;
    let mut iter = SlateDbWalIterator::range(
        1,
        WalIteratorEndBound::Unbounded {
            manifest_reader: Arc::new(FutureManifest),
            poll_interval: Duration::from_millis(10),
            system_clock: Arc::new(slatedb_common::clock::DefaultSystemClock::new()),
        },
        options(),
        store,
    )
    .unwrap();
    let mut first = Box::pin(iter.next());
    loop {
        tokio::select! {
            _ = &mut first => panic!("WAL returned before its payload was released"),
            id = started.recv() => { if id.unwrap() == 1 { break; } }
        }
    }
    let files = gates
        .inner
        .list(None)
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let file = files
        .iter()
        .find(|file| file.location.filename().unwrap() == "00000000000000000001.sst")
        .unwrap();
    gates.inner.delete(&file.location).await.unwrap();
    gates.gates[&1].add_permits(1);
    assert!(matches!(
        tokio::time::timeout(Duration::from_secs(1), first)
            .await
            .unwrap(),
        Err(WalError::WalTruncated(1))
    ));
}
