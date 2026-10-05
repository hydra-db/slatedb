use crate::bytes_range::BytesRange;
use crate::db_state::{SortedRun, SsTableView};
use crate::db_stats::DbStats;
use crate::error::SlateDBError;
use crate::iter::{IterationOrder, RowEntryIterator};
use crate::sst_iter::{SstIterator, SstIteratorOptions, SstView};
use crate::tablestore::TableStore;
use crate::types::RowEntry;
use async_trait::async_trait;
use bytes::Bytes;
use std::collections::VecDeque;
use std::ops::{Bound, RangeBounds};
use std::sync::Arc;

#[derive(Debug)]
enum SortedRunView<'a> {
    Owned(VecDeque<SsTableView>, BytesRange),
    Borrowed(
        VecDeque<&'a SsTableView>,
        (Bound<&'a [u8]>, Bound<&'a [u8]>),
    ),
}

impl<'a> SortedRunView<'a> {
    /// Pops the next table, restricting the iteration range to the table's
    /// view range. Projected tables (e.g. in cloned manifests) may have a
    /// `visible_range` narrower than the requested range; tables whose view
    /// range does not intersect the requested range are skipped entirely.
    fn pop_sst(&mut self, order: IterationOrder) -> Option<SstView<'a>> {
        match self {
            SortedRunView::Owned(tables, r) => loop {
                let table = match order {
                    IterationOrder::Ascending => tables.pop_front(),
                    IterationOrder::Descending => tables.pop_back(),
                }?;
                if let Some(view_range) = table.calculate_view_range(r.clone()) {
                    return Some(SstView::Owned(Box::new(table), view_range));
                }
            },
            SortedRunView::Borrowed(tables, r) => loop {
                let table = match order {
                    IterationOrder::Ascending => tables.pop_front(),
                    IterationOrder::Descending => tables.pop_back(),
                }?;
                if let Some(view_range) = table.calculate_view_range(BytesRange::from_slice(*r)) {
                    return Some(SstView::Borrowed(table, view_range));
                }
            },
        }
    }

    pub(crate) async fn build_next_iter(
        &mut self,
        table_store: Arc<TableStore>,
        sst_iterator_options: SstIteratorOptions,
        db_stats: Option<DbStats>,
    ) -> Result<Option<SstIterator<'a>>, SlateDBError> {
        let next_iter = if let Some(view) = self.pop_sst(sst_iterator_options.order) {
            Some(SstIterator::new_with_stats(
                view,
                table_store,
                sst_iterator_options,
                db_stats,
            )?)
        } else {
            None
        };
        Ok(next_iter)
    }

    fn peek_next_table(&self) -> Option<&SsTableView> {
        match self {
            SortedRunView::Owned(tables, _) => tables.front(),
            SortedRunView::Borrowed(tables, _) => tables.front().copied(),
        }
    }
}

pub(crate) struct SortedRunIterator<'a> {
    table_store: Arc<TableStore>,
    sst_iter_options: SstIteratorOptions,
    db_stats: Option<DbStats>,
    view: SortedRunView<'a>,
    current_iter: Option<SstIterator<'a>>,
    initialized: bool,
    descending_buffer: VecDeque<RowEntry>,
    pending_entry: Option<RowEntry>,
}

impl<'a> SortedRunIterator<'a> {
    async fn new(
        view: SortedRunView<'a>,
        table_store: Arc<TableStore>,
        sst_iter_options: SstIteratorOptions,
        db_stats: Option<DbStats>,
    ) -> Result<Self, SlateDBError> {
        let mut res = Self {
            table_store,
            sst_iter_options,
            db_stats,
            view,
            current_iter: None,
            initialized: false,
            descending_buffer: VecDeque::new(),
            pending_entry: None,
        };
        res.advance_table().await?;
        Ok(res)
    }

    pub(crate) async fn new_owned<T: RangeBounds<Bytes>>(
        range: T,
        sorted_run: SortedRun,
        table_store: Arc<TableStore>,
        sst_iter_options: SstIteratorOptions,
        db_stats: Option<DbStats>,
    ) -> Result<Self, SlateDBError> {
        let range = BytesRange::from(range);
        let tables = sorted_run.into_tables_covering_range(&range);
        let view = SortedRunView::Owned(tables, range);
        SortedRunIterator::new(view, table_store, sst_iter_options, db_stats).await
    }

    #[allow(dead_code)]
    pub(crate) async fn new_owned_initialized<T: RangeBounds<Bytes>>(
        range: T,
        sorted_run: SortedRun,
        table_store: Arc<TableStore>,
        sst_iter_options: SstIteratorOptions,
    ) -> Result<Self, SlateDBError> {
        SortedRunIterator::new_owned_initialized_with_stats(
            range,
            sorted_run,
            table_store,
            sst_iter_options,
            None,
        )
        .await
    }

    pub(crate) async fn new_owned_initialized_with_stats<T: RangeBounds<Bytes>>(
        range: T,
        sorted_run: SortedRun,
        table_store: Arc<TableStore>,
        sst_iter_options: SstIteratorOptions,
        db_stats: Option<DbStats>,
    ) -> Result<Self, SlateDBError> {
        let mut iter = SortedRunIterator::new_owned(
            range,
            sorted_run,
            table_store,
            sst_iter_options,
            db_stats,
        )
        .await?;
        iter.init().await?;
        Ok(iter)
    }

    pub(crate) async fn new_borrowed<T: RangeBounds<&'a [u8]>>(
        range: T,
        sorted_run: &'a SortedRun,
        table_store: Arc<TableStore>,
        sst_iter_options: SstIteratorOptions,
    ) -> Result<Self, SlateDBError> {
        Self::new_borrowed_with_stats(range, sorted_run, table_store, sst_iter_options, None).await
    }

    pub(crate) async fn new_borrowed_with_stats<T: RangeBounds<&'a [u8]>>(
        range: T,
        sorted_run: &'a SortedRun,
        table_store: Arc<TableStore>,
        sst_iter_options: SstIteratorOptions,
        db_stats: Option<DbStats>,
    ) -> Result<Self, SlateDBError> {
        let range = (range.start_bound().cloned(), range.end_bound().cloned());
        let tables = sorted_run.tables_covering_range(BytesRange::from_slice(range));
        let view = SortedRunView::Borrowed(tables, range);
        SortedRunIterator::new(view, table_store, sst_iter_options, db_stats).await
    }

    #[cfg(test)]
    pub(crate) async fn new_borrowed_initialized<T: RangeBounds<&'a [u8]>>(
        range: T,
        sorted_run: &'a SortedRun,
        table_store: Arc<TableStore>,
        sst_iter_options: SstIteratorOptions,
    ) -> Result<Self, SlateDBError> {
        let mut iter =
            SortedRunIterator::new_borrowed(range, sorted_run, table_store, sst_iter_options)
                .await?;
        iter.init().await?;
        Ok(iter)
    }

    async fn advance_table(&mut self) -> Result<(), SlateDBError> {
        self.current_iter = self
            .view
            .build_next_iter(
                self.table_store.clone(),
                self.sst_iter_options.clone(),
                self.db_stats.clone(),
            )
            .await?;
        if self.initialized {
            if let Some(iter) = self.current_iter.as_mut() {
                iter.init().await?;
            }
        }
        Ok(())
    }

    async fn next_raw(&mut self) -> Result<Option<RowEntry>, SlateDBError> {
        while let Some(iter) = &mut self.current_iter {
            if let Some(row) = iter.next().await? {
                return Ok(Some(row));
            }
            self.advance_table().await?;
        }
        Ok(None)
    }

    async fn next_descending(&mut self) -> Result<Option<RowEntry>, SlateDBError> {
        if let Some(row) = self.descending_buffer.pop_front() {
            return Ok(Some(row));
        }
        let first = match self.pending_entry.take() {
            Some(row) => Some(row),
            None => self.next_raw().await?,
        };
        let Some(first) = first else {
            return Ok(None);
        };
        let key = first.key.clone();
        let mut versions = vec![first];
        while let Some(row) = self.next_raw().await? {
            if row.key == key {
                versions.push(row);
            } else {
                self.pending_entry = Some(row);
                break;
            }
        }
        // Historical runs can split a version group across adjacent SSTs.
        // Reversing file traversal must not reverse sequence precedence:
        // snapshot filtering and merge operands still need newest seq first.
        versions.sort_by(|a, b| b.seq.cmp(&a.seq));
        self.descending_buffer.extend(versions);
        Ok(self.descending_buffer.pop_front())
    }
}

#[async_trait]
impl RowEntryIterator for SortedRunIterator<'_> {
    async fn init(&mut self) -> Result<(), SlateDBError> {
        if !self.initialized {
            if let Some(iter) = self.current_iter.as_mut() {
                iter.init().await?;
            }
            self.initialized = true;
        }
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<RowEntry>, SlateDBError> {
        if !self.initialized {
            return Err(SlateDBError::IteratorNotInitialized);
        }
        match self.sst_iter_options.order {
            IterationOrder::Ascending => self.next_raw().await,
            IterationOrder::Descending => self.next_descending().await,
        }
    }

    async fn seek(&mut self, next_key: &[u8]) -> Result<(), SlateDBError> {
        if !self.initialized {
            return Err(SlateDBError::IteratorNotInitialized);
        }
        match self.sst_iter_options.order {
            IterationOrder::Ascending => {
                while let Some(next_table) = self.view.peek_next_table() {
                    if next_table.compacted_effective_start_key() < next_key {
                        self.advance_table().await?;
                    } else {
                        break;
                    }
                }
            }
            IterationOrder::Descending => {
                while self
                    .descending_buffer
                    .front()
                    .is_some_and(|row| row.key.as_ref() > next_key)
                {
                    self.descending_buffer.pop_front();
                }
                if !self.descending_buffer.is_empty()
                    || self
                        .pending_entry
                        .as_ref()
                        .is_some_and(|row| row.key.as_ref() <= next_key)
                {
                    return Ok(());
                }
                self.pending_entry = None;
                // Skip tables whose entire visible range is above the seek
                // key. Comparing the *next* table's start would miss seeks
                // into that table (or into the gap immediately above it).
                while self
                    .current_iter
                    .as_ref()
                    .is_some_and(|iter| iter.key_precedes_view(next_key))
                {
                    self.advance_table().await?;
                }
            }
        }
        if let Some(iter) = &mut self.current_iter {
            iter.seek(next_key).await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_cache_policy::BlockCachePolicy;
    use crate::bytes_generator::OrderedBytesGenerator;
    use crate::db_state::{SsTableHandle, SsTableId};
    use crate::format::sst::SsTableFormat;
    use crate::iter::IterationOrder;
    use crate::proptest_util;
    use crate::proptest_util::sample;
    use crate::tablestore::TableStoreKind;
    use crate::test_utils::assert_kv;
    use crate::types::KeyValue;

    use crate::object_stores::ObjectStores;
    use bytes::{BufMut, BytesMut};
    use object_store::path::Path;
    use object_store::{memory::InMemory, ObjectStore};
    use proptest::test_runner::TestRng;
    use rand::distr::uniform::SampleRange;
    use rand::Rng;
    use std::collections::BTreeMap;
    use std::sync::Arc;

    // Three SSTs with gaps, multiple blocks and multiple versions per key.
    async fn direction_fixture() -> (Arc<TableStore>, SortedRun) {
        let table_store = Arc::new(TableStore::new(
            ObjectStores::new(Arc::new(InMemory::new()), None),
            SsTableFormat {
                block_size: 64,
                ..SsTableFormat::default()
            },
            Path::from("direction-fixture"),
            None,
            TableStoreKind::Main,
            BlockCachePolicy::default(),
        ));
        let mut views = Vec::new();
        for start in [1, 7, 13] {
            let mut builder = table_store.table_builder();
            for number in start..start + 4 {
                let key = format!("key{number:02}");
                for seq in [2, 1] {
                    builder
                        .add(RowEntry::new_value(key.as_bytes(), b"value", seq))
                        .await
                        .unwrap();
                }
            }
            let encoded = builder.build().await.unwrap();
            let handle = table_store
                .write_sst(&SsTableId::Compacted(ulid::Ulid::new()), &encoded)
                .await
                .unwrap();
            views.push(SsTableView::identity(handle));
        }
        (table_store, SortedRun::new(0, views))
    }

    async fn direction_iter<'a>(
        range: &'a BytesRange,
        run: &'a SortedRun,
        store: Arc<TableStore>,
        order: IterationOrder,
        owned: bool,
    ) -> SortedRunIterator<'a> {
        let options = SstIteratorOptions {
            order,
            ..SstIteratorOptions::default()
        };
        if owned {
            SortedRunIterator::new_owned_initialized(range.clone(), run.clone(), store, options)
                .await
                .unwrap()
        } else {
            let bounds = (
                range.start_bound().map(|k| k.as_ref()),
                range.end_bound().map(|k| k.as_ref()),
            );
            SortedRunIterator::new_borrowed_initialized(bounds, run, store, options)
                .await
                .unwrap()
        }
    }

    fn direction_expected(range: &BytesRange, order: IterationOrder) -> Vec<(Bytes, u64)> {
        let mut keys = [1, 2, 3, 4, 7, 8, 9, 10, 13, 14, 15, 16]
            .map(|n| Bytes::from(format!("key{n:02}")))
            .to_vec();
        if matches!(order, IterationOrder::Descending) {
            keys.reverse();
        }
        keys.into_iter()
            .filter(|k| range.contains(k))
            .flat_map(|k| [(k.clone(), 2), (k, 1)])
            .collect()
    }

    async fn direction_drain(iter: &mut SortedRunIterator<'_>) -> Vec<(Bytes, u64)> {
        let mut rows = Vec::new();
        while let Some(row) = iter.next().await.unwrap() {
            rows.push((row.key, row.seq));
        }
        assert!(iter.next().await.unwrap().is_none());
        rows
    }

    #[tokio::test]
    async fn test_sorted_run_direction_ranges_and_versions() {
        let (store, run) = direction_fixture().await;
        let ranges = [
            BytesRange::from(..),
            BytesRange::from_ref("key04"..="key13"),
            BytesRange::from_ref("key04".."key13"),
            BytesRange::new(
                Bound::Excluded(Bytes::from("key04")),
                Bound::Excluded(Bytes::from("key13")),
            ),
            BytesRange::from_ref("key07"..="key07"),
            BytesRange::from_ref("key05"..="key06"),
            BytesRange::from_ref("key00"..="key00"),
            BytesRange::from_ref("key18"..),
        ];
        for range in &ranges {
            for order in [IterationOrder::Ascending, IterationOrder::Descending] {
                for owned in [true, false] {
                    let mut iter = direction_iter(range, &run, store.clone(), order, owned).await;
                    assert_eq!(
                        direction_drain(&mut iter).await,
                        direction_expected(range, order),
                        "range={range:?}, order={order:?}, owned={owned}"
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn test_sorted_run_direction_seek_into_tables_gaps_and_bounds() {
        let (store, run) = direction_fixture().await;
        let range = BytesRange::from_ref("key02"..="key15");
        for order in [IterationOrder::Ascending, IterationOrder::Descending] {
            for owned in [true, false] {
                // Stay inside the requested range, but include SST gaps and
                // exact file-start/file-end keys. A seek must be inclusive.
                for number in 2..=15 {
                    let key = Bytes::from(format!("key{number:02}"));
                    let mut iter = direction_iter(&range, &run, store.clone(), order, owned).await;
                    iter.seek(&key).await.unwrap();
                    let expected = direction_expected(&range, order)
                        .into_iter()
                        .filter(|(k, _)| match order {
                            IterationOrder::Ascending => k >= &key,
                            IterationOrder::Descending => k <= &key,
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(
                        direction_drain(&mut iter).await,
                        expected,
                        "seek={key:?}, order={order:?}, owned={owned}"
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn test_sorted_run_descending_seek_after_read_ahead() {
        let (store, run) = direction_fixture().await;
        let range = BytesRange::from(..);
        for owned in [true, false] {
            let mut iter = direction_iter(
                &range,
                &run,
                store.clone(),
                IterationOrder::Descending,
                owned,
            )
            .await;
            let first = iter.next().await.unwrap().unwrap();
            assert_eq!((first.key, first.seq), (Bytes::from("key16"), 2));
            // Preserve the unconsumed version of the same key, then skip the
            // next buffered group, cross an SST gap and seek below the run.
            iter.seek(b"key16").await.unwrap();
            let second = iter.next().await.unwrap().unwrap();
            assert_eq!((second.key, second.seq), (Bytes::from("key16"), 1));
            for key in ["key14", "key11", "key08", "key05", "key01"] {
                iter.seek(key.as_bytes()).await.unwrap();
                let row = iter.next().await.unwrap().unwrap();
                let expected_key = direction_expected(&range, IterationOrder::Descending)
                    .into_iter()
                    .find(|(k, _)| k.as_ref() <= key.as_bytes())
                    .unwrap()
                    .0;
                assert_eq!((row.key, row.seq), (expected_key, 2));
            }
            iter.seek(b"key00").await.unwrap();
            assert!(iter.next().await.unwrap().is_none());
        }
    }

    #[tokio::test]
    async fn test_sorted_run_descending_projected_ranges() {
        let (store, run) = direction_fixture().await;
        let views = run
            .sst_views()
            .iter()
            .zip([("key02", "key04"), ("key08", "key10"), ("key14", "key16")])
            .map(|(view, (lo, hi))| {
                SsTableView::new_projected(
                    ulid::Ulid::new(),
                    view.sst.clone(),
                    Some(BytesRange::from_ref(lo..hi)),
                )
            })
            .collect::<Vec<_>>();
        let projected = SortedRun::new(0, views);
        let range = BytesRange::from_ref("key03"..="key14");
        let expected = ["key14", "key09", "key08", "key03"]
            .into_iter()
            .flat_map(|k| [(Bytes::from(k), 2), (Bytes::from(k), 1)])
            .collect::<Vec<_>>();
        for owned in [true, false] {
            let mut iter = direction_iter(
                &range,
                &projected,
                store.clone(),
                IterationOrder::Descending,
                owned,
            )
            .await;
            assert_eq!(direction_drain(&mut iter).await, expected);
            let mut iter = direction_iter(
                &range,
                &projected,
                store.clone(),
                IterationOrder::Descending,
                owned,
            )
            .await;
            iter.seek(b"key13").await.unwrap();
            assert_eq!(direction_drain(&mut iter).await, expected[2..]);
        }
    }

    #[tokio::test]
    async fn test_sorted_run_empty_and_uninitialized_descending() {
        let (store, _) = direction_fixture().await;
        let run = SortedRun::new(0, []);
        for owned in [true, false] {
            let range = BytesRange::from(..);
            let mut iter = direction_iter(
                &range,
                &run,
                store.clone(),
                IterationOrder::Descending,
                owned,
            )
            .await;
            iter.seek(b"key08").await.unwrap();
            assert!(direction_drain(&mut iter).await.is_empty());
        }
        let mut iter = SortedRunIterator::new_owned(
            ..,
            run,
            store,
            SstIteratorOptions {
                order: IterationOrder::Descending,
                ..SstIteratorOptions::default()
            },
            None,
        )
        .await
        .unwrap();
        assert!(matches!(
            iter.seek(b"key08").await,
            Err(SlateDBError::IteratorNotInitialized)
        ));
        assert!(matches!(
            iter.next().await,
            Err(SlateDBError::IteratorNotInitialized)
        ));
    }

    #[tokio::test]
    async fn test_sorted_run_descending_seek_without_last_entry_metadata() {
        let (store, run) = direction_fixture().await;
        let run = SortedRun::new(
            0,
            run.sst_views().iter().map(|view| {
                let mut handle = view.sst.clone();
                handle.info.last_entry = None;
                SsTableView::identity(handle)
            }),
        );
        let range = BytesRange::from(..);
        for owned in [true, false] {
            for target in ["key14", "key11", "key07", "key05", "key00"] {
                let mut iter = direction_iter(
                    &range,
                    &run,
                    store.clone(),
                    IterationOrder::Descending,
                    owned,
                )
                .await;
                iter.seek(target.as_bytes()).await.unwrap();
                let expected = direction_expected(&range, IterationOrder::Descending)
                    .into_iter()
                    .filter(|(k, _)| k.as_ref() <= target.as_bytes())
                    .collect::<Vec<_>>();
                assert_eq!(direction_drain(&mut iter).await, expected);
            }
        }
    }

    #[tokio::test]
    async fn test_sorted_run_descending_versions_at_shared_sst_boundary() {
        let (store, _) = direction_fixture().await;
        let mut views = Vec::new();
        for rows in [
            vec![("a", 1), ("b", 5), ("b", 4)],
            vec![("b", 3), ("b", 2)],
            vec![("b", 1), ("c", 1)],
        ] {
            let mut builder = store.table_builder();
            for (key, seq) in rows {
                builder
                    .add(RowEntry::new_value(
                        key.as_bytes(),
                        format!("value{seq}").as_bytes(),
                        seq,
                    ))
                    .await
                    .unwrap();
            }
            let encoded = builder.build().await.unwrap();
            views.push(SsTableView::identity(
                store
                    .write_sst(&SsTableId::Compacted(ulid::Ulid::new()), &encoded)
                    .await
                    .unwrap(),
            ));
        }
        let run = SortedRun::new(0, views);
        let expected = [
            ("c", 1),
            ("b", 5),
            ("b", 4),
            ("b", 3),
            ("b", 2),
            ("b", 1),
            ("a", 1),
        ]
        .map(|(k, seq)| (Bytes::from(k), seq))
        .to_vec();
        let range = BytesRange::from(..);
        for owned in [true, false] {
            let mut iter = direction_iter(
                &range,
                &run,
                store.clone(),
                IterationOrder::Descending,
                owned,
            )
            .await;
            assert_eq!(direction_drain(&mut iter).await, expected);
            let point = BytesRange::from_ref("b"..="b");
            let mut iter = direction_iter(
                &point,
                &run,
                store.clone(),
                IterationOrder::Descending,
                owned,
            )
            .await;
            assert_eq!(direction_drain(&mut iter).await, expected[1..6]);
            let mut iter = direction_iter(
                &range,
                &run,
                store.clone(),
                IterationOrder::Descending,
                owned,
            )
            .await;
            assert_eq!(iter.next().await.unwrap().unwrap().key.as_ref(), b"c");
            iter.seek(b"b").await.unwrap();
            assert_eq!(iter.next().await.unwrap().unwrap().seq, 5);
            iter.seek(b"b").await.unwrap();
            assert_eq!(iter.next().await.unwrap().unwrap().seq, 4);
            iter.seek(b"a").await.unwrap();
            assert_eq!(direction_drain(&mut iter).await, expected[6..]);
        }
        // Verify the public iterator's snapshot filter/dedup sees the correct
        // version, rather than retaining the oldest SST's boundary entry.
        for max_seq in [None, Some(3)] {
            let raw = SortedRunIterator::new_owned_initialized(
                ..,
                run.clone(),
                store.clone(),
                SstIteratorOptions {
                    order: IterationOrder::Descending,
                    ..SstIteratorOptions::default()
                },
            )
            .await
            .unwrap();
            let mut iter = crate::db_iter::DbIterator::new(
                range.clone(),
                None,
                Vec::<Box<dyn RowEntryIterator>>::new(),
                Box::new(raw),
                max_seq,
                None,
                IterationOrder::Descending,
            )
            .await
            .unwrap();
            assert_eq!(iter.next().await.unwrap().unwrap().key.as_ref(), b"c");
            let row = iter.next().await.unwrap().unwrap();
            assert_eq!(row.key.as_ref(), b"b");
            assert_eq!(
                row.value.as_ref(),
                if max_seq.is_some() {
                    b"value3"
                } else {
                    b"value5"
                }
            );
            assert_eq!(iter.next().await.unwrap().unwrap().key.as_ref(), b"a");
            assert!(iter.next().await.unwrap().is_none());
        }
    }

    #[tokio::test]
    async fn test_one_sst_sr_iter() {
        let root_path = Path::from("");
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let format = SsTableFormat {
            min_filter_keys: 3,
            ..SsTableFormat::default()
        };
        let table_store = Arc::new(TableStore::new(
            ObjectStores::new(object_store, None),
            format,
            root_path.clone(),
            None,
            TableStoreKind::Main,
            BlockCachePolicy::default(),
        ));
        let mut builder = table_store.table_builder();
        builder
            .add_value(b"key1", b"value1", Some(1), None)
            .await
            .unwrap();
        builder
            .add_value(b"key2", b"value2", Some(2), None)
            .await
            .unwrap();
        builder
            .add_value(b"key3", b"value3", Some(3), None)
            .await
            .unwrap();
        let encoded = builder.build().await.unwrap();
        let id = SsTableId::Compacted(ulid::Ulid::new());
        let handle = table_store.write_sst(&id, &encoded).await.unwrap();
        let sr = SortedRun::new(0, [SsTableView::identity(handle)]);

        let mut iter = SortedRunIterator::new_owned_initialized(
            ..,
            sr,
            table_store,
            SstIteratorOptions::default(),
        )
        .await
        .unwrap();

        let kv: KeyValue = iter.next().await.unwrap().unwrap().into();
        assert_eq!(kv.key, b"key1".as_slice());
        assert_eq!(kv.value, b"value1".as_slice());
        let kv: KeyValue = iter.next().await.unwrap().unwrap().into();
        assert_eq!(kv.key, b"key2".as_slice());
        assert_eq!(kv.value, b"value2".as_slice());
        let kv: KeyValue = iter.next().await.unwrap().unwrap().into();
        assert_eq!(kv.key, b"key3".as_slice());
        assert_eq!(kv.value, b"value3".as_slice());
        let kv = iter.next().await.unwrap().map(KeyValue::from);
        assert!(kv.is_none());
    }

    #[tokio::test]
    async fn test_many_sst_sr_iter() {
        let root_path = Path::from("");
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let format = SsTableFormat {
            min_filter_keys: 3,
            ..SsTableFormat::default()
        };
        let table_store = Arc::new(TableStore::new(
            ObjectStores::new(object_store, None),
            format,
            root_path.clone(),
            None,
            TableStoreKind::Main,
            BlockCachePolicy::default(),
        ));
        let mut builder = table_store.table_builder();
        builder
            .add_value(b"key1", b"value1", Some(1), None)
            .await
            .unwrap();
        builder
            .add_value(b"key2", b"value2", Some(2), None)
            .await
            .unwrap();
        let encoded = builder.build().await.unwrap();
        let id1 = SsTableId::Compacted(ulid::Ulid::new());
        let handle1 = table_store.write_sst(&id1, &encoded).await.unwrap();
        let mut builder = table_store.table_builder();
        builder
            .add_value(b"key3", b"value3", Some(3), None)
            .await
            .unwrap();
        let encoded = builder.build().await.unwrap();
        let id2 = SsTableId::Compacted(ulid::Ulid::new());
        let handle2 = table_store.write_sst(&id2, &encoded).await.unwrap();
        let sr = SortedRun::new(
            0,
            [
                SsTableView::identity(handle1),
                SsTableView::identity(handle2),
            ],
        );

        let mut iter = SortedRunIterator::new_owned_initialized(
            ..,
            sr,
            table_store.clone(),
            SstIteratorOptions::default(),
        )
        .await
        .unwrap();

        let kv: KeyValue = iter.next().await.unwrap().unwrap().into();
        assert_eq!(kv.key, b"key1".as_slice());
        assert_eq!(kv.value, b"value1".as_slice());
        let kv: KeyValue = iter.next().await.unwrap().unwrap().into();
        assert_eq!(kv.key, b"key2".as_slice());
        assert_eq!(kv.value, b"value2".as_slice());
        let kv: KeyValue = iter.next().await.unwrap().unwrap().into();
        assert_eq!(kv.key, b"key3".as_slice());
        assert_eq!(kv.value, b"value3".as_slice());
        let kv = iter.next().await.unwrap().map(KeyValue::from);
        assert!(kv.is_none());
    }

    #[tokio::test]
    async fn test_sr_iter_respects_visible_range() {
        // given: a sorted run whose views carry visible_range restrictions,
        // as produced by manifest projection (e.g. range-restricted clones)
        let root_path = Path::from("");
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let format = SsTableFormat {
            min_filter_keys: 3,
            ..SsTableFormat::default()
        };
        let table_store = Arc::new(TableStore::new(
            ObjectStores::new(object_store, None),
            format,
            root_path.clone(),
            None,
            TableStoreKind::Main,
            BlockCachePolicy::default(),
        ));
        let mut builder = table_store.table_builder();
        for i in 1..=4 {
            let key = format!("key{i}");
            let value = format!("value{i}");
            builder
                .add_value(key.as_bytes(), value.as_bytes(), Some(i), None)
                .await
                .unwrap();
        }
        let encoded = builder.build().await.unwrap();
        let id1 = SsTableId::Compacted(ulid::Ulid::new());
        let handle1 = table_store.write_sst(&id1, &encoded).await.unwrap();
        let mut builder = table_store.table_builder();
        for i in 5..=8 {
            let key = format!("key{i}");
            let value = format!("value{i}");
            builder
                .add_value(key.as_bytes(), value.as_bytes(), Some(i), None)
                .await
                .unwrap();
        }
        let encoded = builder.build().await.unwrap();
        let id2 = SsTableId::Compacted(ulid::Ulid::new());
        let handle2 = table_store.write_sst(&id2, &encoded).await.unwrap();
        let sr = SortedRun::new(
            0,
            [
                SsTableView::new_projected(
                    ulid::Ulid::new(),
                    handle1,
                    Some(BytesRange::from_ref("key2".."key4")),
                ),
                SsTableView::new_projected(
                    ulid::Ulid::new(),
                    handle2,
                    Some(BytesRange::from_ref("key5".."key7")),
                ),
            ],
        );

        // when: iterating the full range, then: only visible keys appear
        let mut iter = SortedRunIterator::new_borrowed_initialized(
            ..,
            &sr,
            table_store.clone(),
            SstIteratorOptions::default(),
        )
        .await
        .unwrap();
        for i in [2, 3, 5, 6] {
            let kv: KeyValue = iter.next().await.unwrap().unwrap().into();
            assert_eq!(kv.key.as_ref(), format!("key{i}").as_bytes());
        }
        assert!(iter.next().await.unwrap().is_none());

        // when: iterating a sub-range, then: the narrower bound applies and
        // tables disjoint with the query range are skipped entirely
        let mut iter = SortedRunIterator::new_owned_initialized(
            BytesRange::from_ref("key6"..),
            sr,
            table_store.clone(),
            SstIteratorOptions::default(),
        )
        .await
        .unwrap();
        let kv: KeyValue = iter.next().await.unwrap().unwrap().into();
        assert_eq!(kv.key.as_ref(), b"key6");
        assert!(iter.next().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_sr_iter_from_key() {
        let root_path = Path::from("");
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let format = SsTableFormat {
            min_filter_keys: 3,
            ..SsTableFormat::default()
        };
        let table_store = Arc::new(TableStore::new(
            ObjectStores::new(object_store, None),
            format,
            root_path.clone(),
            None,
            TableStoreKind::Main,
            BlockCachePolicy::default(),
        ));
        let key_gen = OrderedBytesGenerator::new_with_byte_range(&[b'a'; 16], b'a', b'z');
        let mut test_case_key_gen = key_gen.clone();
        let val_gen = OrderedBytesGenerator::new_with_byte_range(&[0u8; 16], 0u8, 26u8);
        let mut test_case_val_gen = val_gen.clone();
        let sr = build_sr_with_ssts(table_store.clone(), 3, 10, key_gen, val_gen).await;

        for i in 0..30 {
            let mut expected_key_gen = test_case_key_gen.clone();
            let mut expected_val_gen = test_case_val_gen.clone();
            let from_key = test_case_key_gen.next();
            _ = test_case_val_gen.next();
            let mut iter = SortedRunIterator::new_borrowed_initialized(
                from_key.as_ref()..,
                &sr,
                table_store.clone(),
                SstIteratorOptions::default(),
            )
            .await
            .unwrap();
            for _ in 0..30 - i {
                assert_kv(
                    &iter.next().await.unwrap().unwrap().into(),
                    expected_key_gen.next().as_ref(),
                    expected_val_gen.next().as_ref(),
                );
            }
            assert!(iter.next().await.unwrap().is_none());
        }
    }

    #[tokio::test]
    async fn test_sr_iter_from_key_lower_than_range() {
        let root_path = Path::from("");
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let format = SsTableFormat {
            min_filter_keys: 3,
            ..SsTableFormat::default()
        };
        let table_store = Arc::new(TableStore::new(
            ObjectStores::new(object_store, None),
            format,
            root_path.clone(),
            None,
            TableStoreKind::Main,
            BlockCachePolicy::default(),
        ));
        let key_gen = OrderedBytesGenerator::new_with_byte_range(&[b'a'; 16], b'a', b'z');
        let mut expected_key_gen = key_gen.clone();
        let val_gen = OrderedBytesGenerator::new_with_byte_range(&[0u8; 16], 0u8, 26u8);
        let mut expected_val_gen = val_gen.clone();
        let sr = build_sr_with_ssts(table_store.clone(), 3, 10, key_gen, val_gen).await;
        let mut iter = SortedRunIterator::new_borrowed_initialized(
            [b'a', 10].as_ref()..,
            &sr,
            table_store.clone(),
            SstIteratorOptions::default(),
        )
        .await
        .unwrap();

        for _ in 0..30 {
            assert_kv(
                &iter.next().await.unwrap().unwrap().into(),
                expected_key_gen.next().as_ref(),
                expected_val_gen.next().as_ref(),
            );
        }
        assert!(iter.next().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_sr_iter_from_key_higher_than_range() {
        let root_path = Path::from("");
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let format = SsTableFormat {
            min_filter_keys: 3,
            ..SsTableFormat::default()
        };
        let table_store = Arc::new(TableStore::new(
            ObjectStores::new(object_store, None),
            format,
            root_path.clone(),
            None,
            TableStoreKind::Main,
            BlockCachePolicy::default(),
        ));
        let key_gen = OrderedBytesGenerator::new_with_byte_range(&[b'a'; 16], b'a', b'z');
        let val_gen = OrderedBytesGenerator::new_with_byte_range(&[0u8; 16], 0u8, 26u8);
        let sr = build_sr_with_ssts(table_store.clone(), 3, 10, key_gen, val_gen).await;

        let mut iter = SortedRunIterator::new_borrowed_initialized(
            [b'z', 30].as_ref()..,
            &sr,
            table_store.clone(),
            SstIteratorOptions::default(),
        )
        .await
        .unwrap();

        assert!(iter.next().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_seek_through_sorted_run() {
        let root_path = Path::from("");
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let table_store = Arc::new(TableStore::new(
            ObjectStores::new(object_store, None),
            SsTableFormat::default(),
            root_path.clone(),
            None,
            TableStoreKind::Main,
            BlockCachePolicy::default(),
        ));

        let mut rng = proptest_util::rng::new_test_rng(None);
        let table = sample::table(&mut rng, 400, 10);
        let max_entries_per_sst = 20u64;
        let entries_per_sst = 1..max_entries_per_sst;
        let sr =
            build_sorted_run_from_table(&table, table_store.clone(), entries_per_sst, &mut rng)
                .await;
        let mut sr_iter = SortedRunIterator::new_owned_initialized(
            ..,
            sr,
            table_store.clone(),
            SstIteratorOptions::default(),
        )
        .await
        .unwrap();
        let mut table_iter = table.iter();
        loop {
            let skip = rng.random::<u64>() % (max_entries_per_sst * 2);
            let run = rng.random::<u64>() % (max_entries_per_sst * 2);

            let Some((k, _)) = table_iter.nth(skip as usize) else {
                break;
            };
            let seek_key = increment_length(k);
            sr_iter.seek(&seek_key).await.unwrap();

            for (key, value) in table_iter.by_ref().take(run as usize) {
                let kv: KeyValue = sr_iter.next().await.unwrap().unwrap().into();
                assert_eq!(*key, kv.key);
                assert_eq!(*value, kv.value);
            }
        }
    }

    fn increment_length(b: &[u8]) -> Bytes {
        let mut buf = BytesMut::from(b);
        buf.put_u8(u8::MIN);
        buf.freeze()
    }

    async fn build_sorted_run_from_table<R: SampleRange<u64> + Clone>(
        table: &BTreeMap<Bytes, Bytes>,
        table_store: Arc<TableStore>,
        entries_per_sst: R,
        rng: &mut TestRng,
    ) -> SortedRun {
        let mut ssts = Vec::new();
        let mut entries = table.iter();
        loop {
            let sst_len = rng.random_range(entries_per_sst.clone());
            let mut builder = table_store.table_builder();

            let sst_kvs: Vec<(&Bytes, &Bytes)> = entries.by_ref().take(sst_len as usize).collect();
            if sst_kvs.is_empty() {
                break;
            }

            for (key, value) in sst_kvs {
                builder.add_value(key, value, Some(0), None).await.unwrap();
            }

            let encoded = builder.build().await.unwrap();
            let id = SsTableId::Compacted(ulid::Ulid::new());
            let handle = table_store.write_sst(&id, &encoded).await.unwrap();
            ssts.push(SsTableView::identity(handle));
        }

        SortedRun::new(0, ssts)
    }

    async fn build_sr_with_ssts(
        table_store: Arc<TableStore>,
        n: usize,
        keys_per_sst: usize,
        mut key_gen: OrderedBytesGenerator,
        mut val_gen: OrderedBytesGenerator,
    ) -> SortedRun {
        let mut ssts = Vec::<SsTableView>::new();
        for _ in 0..n {
            let mut writer = table_store.table_writer(SsTableId::Compacted(ulid::Ulid::new()));
            for _ in 0..keys_per_sst {
                let entry =
                    RowEntry::new_value(key_gen.next().as_ref(), val_gen.next().as_ref(), 0);
                writer.add(entry).await.unwrap();
            }
            let sst = writer.close().await.unwrap();
            ssts.push(SsTableView::identity(sst));
        }
        SortedRun::new(0, ssts)
    }

    mod mixed_version_tests {
        use super::*;
        use crate::sst_builder::BlockFormat;

        async fn build_sst_v1(
            table_store: &Arc<TableStore>,
            keys_and_values: &[(&[u8], &[u8])],
        ) -> SsTableHandle {
            let mut builder = table_store
                .table_builder()
                .with_block_format(BlockFormat::V1);
            for (key, value) in keys_and_values {
                builder.add_value(key, value, Some(0), None).await.unwrap();
            }
            let encoded = builder.build().await.unwrap();
            let id = SsTableId::Compacted(ulid::Ulid::new());
            table_store.write_sst(&id, &encoded).await.unwrap()
        }

        async fn build_sst_v2(
            table_store: &Arc<TableStore>,
            keys_and_values: &[(&[u8], &[u8])],
        ) -> SsTableHandle {
            // V2 is now the default, so no need to explicitly set block format
            let mut builder = table_store.table_builder();
            for (key, value) in keys_and_values {
                builder.add_value(key, value, Some(0), None).await.unwrap();
            }
            let encoded = builder.build().await.unwrap();
            let id = SsTableId::Compacted(ulid::Ulid::new());
            table_store.write_sst(&id, &encoded).await.unwrap()
        }

        #[tokio::test]
        async fn should_iterate_sorted_run_with_mixed_v1_and_v2_ssts() {
            // given: a sorted run with alternating v1 and v2 SSTs
            let root_path = Path::from("");
            let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
            let format = SsTableFormat {
                min_filter_keys: 10,
                ..SsTableFormat::default()
            };
            let table_store = Arc::new(TableStore::new(
                ObjectStores::new(object_store, None),
                format,
                root_path,
                None,
                TableStoreKind::Main,
                BlockCachePolicy::default(),
            ));

            // Build a sorted run with v1, v2, v1, v2 SSTs
            let sst1_v1 = build_sst_v1(
                &table_store,
                &[(b"key01", b"value01"), (b"key02", b"value02")],
            )
            .await;
            let sst2_v2 = build_sst_v2(
                &table_store,
                &[(b"key03", b"value03"), (b"key04", b"value04")],
            )
            .await;
            let sst3_v1 = build_sst_v1(
                &table_store,
                &[(b"key05", b"value05"), (b"key06", b"value06")],
            )
            .await;
            let sst4_v2 = build_sst_v2(
                &table_store,
                &[(b"key07", b"value07"), (b"key08", b"value08")],
            )
            .await;

            let sorted_run = SortedRun::new(
                0,
                [
                    SsTableView::identity(sst1_v1),
                    SsTableView::identity(sst2_v2),
                    SsTableView::identity(sst3_v1),
                    SsTableView::identity(sst4_v2),
                ],
            );

            // when: iterating over the sorted run
            let mut iter = SortedRunIterator::new_owned_initialized(
                ..,
                sorted_run,
                table_store.clone(),
                SstIteratorOptions::default(),
            )
            .await
            .unwrap();

            // then: all keys should be returned in order across both v1 and v2 SSTs
            for i in 1..=8 {
                let kv: KeyValue = iter.next().await.unwrap().unwrap().into();
                let expected_key = format!("key{:02}", i);
                let expected_value = format!("value{:02}", i);
                assert_eq!(kv.key.as_ref(), expected_key.as_bytes());
                assert_eq!(kv.value.as_ref(), expected_value.as_bytes());
            }

            let kv = iter.next().await.unwrap().map(KeyValue::from);
            assert!(kv.is_none());
        }

        #[tokio::test]
        async fn should_seek_through_mixed_v1_and_v2_ssts() {
            // given: a sorted run with alternating v1 and v2 SSTs
            let root_path = Path::from("");
            let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
            let format = SsTableFormat {
                min_filter_keys: 10,
                ..SsTableFormat::default()
            };
            let table_store = Arc::new(TableStore::new(
                ObjectStores::new(object_store, None),
                format,
                root_path,
                None,
                TableStoreKind::Main,
                BlockCachePolicy::default(),
            ));

            // Build a sorted run with v1, v2, v1, v2 SSTs
            let sst1_v1 = build_sst_v1(
                &table_store,
                &[(b"key01", b"value01"), (b"key02", b"value02")],
            )
            .await;
            let sst2_v2 = build_sst_v2(
                &table_store,
                &[(b"key03", b"value03"), (b"key04", b"value04")],
            )
            .await;
            let sst3_v1 = build_sst_v1(
                &table_store,
                &[(b"key05", b"value05"), (b"key06", b"value06")],
            )
            .await;
            let sst4_v2 = build_sst_v2(
                &table_store,
                &[(b"key07", b"value07"), (b"key08", b"value08")],
            )
            .await;

            let sorted_run = SortedRun::new(
                0,
                [
                    SsTableView::identity(sst1_v1),
                    SsTableView::identity(sst2_v2),
                    SsTableView::identity(sst3_v1),
                    SsTableView::identity(sst4_v2),
                ],
            );

            let mut iter = SortedRunIterator::new_owned_initialized(
                ..,
                sorted_run,
                table_store.clone(),
                SstIteratorOptions::default(),
            )
            .await
            .unwrap();

            // when: seeking to key05 (which is in a v1 SST after a v2 SST)
            iter.seek(b"key05").await.unwrap();

            // then: we should get key05 and subsequent keys
            let kv: KeyValue = iter.next().await.unwrap().unwrap().into();
            assert_eq!(kv.key.as_ref(), b"key05");
            assert_eq!(kv.value.as_ref(), b"value05");

            let kv: KeyValue = iter.next().await.unwrap().unwrap().into();
            assert_eq!(kv.key.as_ref(), b"key06");
            assert_eq!(kv.value.as_ref(), b"value06");

            // Seek again to a v2 SST
            iter.seek(b"key07").await.unwrap();

            let kv: KeyValue = iter.next().await.unwrap().unwrap().into();
            assert_eq!(kv.key.as_ref(), b"key07");
            assert_eq!(kv.value.as_ref(), b"value07");
        }
    }
}
