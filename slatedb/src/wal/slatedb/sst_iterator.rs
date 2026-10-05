#![allow(dead_code)] // Implemented ahead of migrating WAL replay to WalTableStore.

use std::collections::VecDeque;
use std::sync::Arc;

use super::store::{PrefetchedWalBlocks, WalFileHandle, WalTableStore};
use crate::block_iterator::DataBlockIterator;
use crate::config::SstBlockSize;
use crate::error::SlateDBError;
use crate::flatbuffer_types::SsTableIndexOwned;
use crate::format::block::Block;
use crate::iter::IterationOrder;
use crate::types::RowEntry;

#[derive(Clone, Debug)]
pub(crate) struct WalSstIteratorOptions {
    /// Target encoded bytes per block-fetch request.
    pub(crate) target_bytes_to_fetch: usize,
}

impl Default for WalSstIteratorOptions {
    fn default() -> Self {
        Self {
            target_bytes_to_fetch: SstBlockSize::default().as_bytes(),
        }
    }
}

/// Iterates all rows in one WAL SST in ascending sequence order.
///
/// WAL replay only performs whole-file scans, so this iterator deliberately has
/// no range/view abstraction, filters, cache controls, descending mode, seek,
/// or independent speculative fetch scheduling. The WAL-file queue can
/// prefetch a bounded encoded prefix; only the current file is decompressed.
pub(crate) struct WalSstIterator {
    table: WalFileHandle,
    index: Arc<SsTableIndexOwned>,
    block_iter: Option<DataBlockIterator<Arc<Block>>>,
    next_block_idx_to_fetch: usize,
    fetched_blocks: VecDeque<Arc<Block>>,
    table_store: Arc<WalTableStore>,
    options: WalSstIteratorOptions,
    prefetched: Option<PrefetchedWalBlocks>,
}

impl WalSstIterator {
    pub(crate) async fn new(
        table: WalFileHandle,
        table_store: Arc<WalTableStore>,
        options: WalSstIteratorOptions,
    ) -> Result<Self, SlateDBError> {
        assert!(options.target_bytes_to_fetch > 0);
        let index = table_store.read_index(&table).await?;
        Ok(Self {
            table,
            index,
            block_iter: None,
            next_block_idx_to_fetch: 0,
            fetched_blocks: VecDeque::new(),
            table_store,
            options,
            prefetched: None,
        })
    }

    pub(crate) async fn prefetch_initial_blocks(
        &mut self,
        max_bytes: usize,
    ) -> Result<(), SlateDBError> {
        self.prefetched = self
            .table_store
            .prefetch_initial_blocks(
                &self.table,
                &self.index,
                max_bytes.min(self.options.target_bytes_to_fetch),
            )
            .await?;
        Ok(())
    }

    /// Returns the next WAL row in ascending sequence order.
    pub(crate) async fn next(&mut self) -> Result<Option<RowEntry>, SlateDBError> {
        loop {
            if let Some(iter) = &mut self.block_iter {
                if let Some(row) = iter.next().await? {
                    return Ok(Some(row));
                }
            }
            if !self.load_next_block().await? {
                return Ok(None);
            }
        }
    }

    async fn load_next_block(&mut self) -> Result<bool, SlateDBError> {
        loop {
            if let Some(block) = self.fetched_blocks.pop_front() {
                self.block_iter = Some(DataBlockIterator::new(
                    block,
                    self.table.format_version,
                    IterationOrder::Ascending,
                )?);
                return Ok(true);
            }

            let num_blocks = self.index.borrow().block_meta().len();
            if self.next_block_idx_to_fetch == num_blocks {
                self.block_iter = None;
                return Ok(false);
            }

            let blocks = match &self.prefetched {
                Some(prefetched) => prefetched.blocks.clone(),
                None => self.table_store.block_range_for_target_bytes(
                    &self.table,
                    &self.index,
                    self.next_block_idx_to_fetch,
                    self.options.target_bytes_to_fetch,
                ),
            };
            let next_block_idx_to_fetch = blocks.end;
            let fetched_blocks = self
                .table_store
                .read_blocks_with_prefetch(
                    &self.table,
                    Arc::clone(&self.index),
                    blocks,
                    self.prefetched.as_ref(),
                )
                .await?;
            // Commit the cursor only after the read succeeds so cancelling the
            // read future cannot cause the next call to skip these blocks.
            self.next_block_idx_to_fetch = next_block_idx_to_fetch;
            self.fetched_blocks = fetched_blocks;
            self.prefetched = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use object_store::memory::InMemory;
    use object_store::path::Path;
    use object_store::ObjectStore;

    use super::*;
    use crate::flatbuffer_types::FlatBufferSsTableInfoCodec;
    use crate::format::sst::SsTableFormat;
    use crate::object_store_tag::TableStoreKind;
    use crate::wal::slatedb::sst_builder::EncodedWalSsTableBuilder;

    fn test_store() -> Arc<WalTableStore> {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        Arc::new(WalTableStore::new(
            object_store,
            SsTableFormat::default(),
            Path::from("test-db"),
            TableStoreKind::Main,
        ))
    }

    #[tokio::test]
    async fn should_iterate_the_whole_wal_sequentially() {
        let store = test_store();
        let rows: Vec<_> = (1..=6)
            .map(|seq| {
                RowEntry::new_value(
                    format!("key-{seq}").as_bytes(),
                    format!("value-{seq}").as_bytes(),
                    seq,
                )
            })
            .collect();
        let mut builder =
            EncodedWalSsTableBuilder::new(32, Box::new(FlatBufferSsTableInfoCodec {}));
        for row in rows.iter().cloned() {
            builder.add(row).await.unwrap();
        }
        let encoded = builder.build().await.unwrap();
        let table = store.write_sst(1, &encoded).await.unwrap();
        let mut iter = WalSstIterator::new(
            table,
            store,
            WalSstIteratorOptions {
                target_bytes_to_fetch: 1,
            },
        )
        .await
        .unwrap();

        let mut actual = Vec::new();
        while let Some(row) = iter.next().await.unwrap() {
            actual.push(row);
        }
        assert_eq!(actual, rows);
        assert!(iter.next().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn bounded_prefix_prefetch_keeps_the_rest_of_a_large_wal() {
        let store = test_store();
        let mut builder =
            EncodedWalSsTableBuilder::new(64, Box::new(FlatBufferSsTableInfoCodec {}));
        let rows = (1..=20)
            .map(|seq| RowEntry::new_value(format!("key-{seq:02}").as_bytes(), &[b'x'; 64], seq))
            .collect::<Vec<_>>();
        for row in rows.iter().cloned() {
            builder.add(row).await.unwrap();
        }
        let encoded = builder.build().await.unwrap();
        let table = store.write_sst(1, &encoded).await.unwrap();
        let mut iter = WalSstIterator::new(
            table.clone(),
            store.clone(),
            WalSstIteratorOptions {
                target_bytes_to_fetch: 1024 * 1024,
            },
        )
        .await
        .unwrap();
        let first_block_size = store.block_range_size(&table, &iter.index, 0..1);
        let cap = first_block_size * 2;
        iter.prefetch_initial_blocks(cap).await.unwrap();
        let prefix = iter.prefetched.as_ref().unwrap();
        assert!(store.block_range_size(&table, &iter.index, prefix.blocks.clone()) <= cap);
        assert!(prefix.blocks.end < iter.index.borrow().block_meta().len());
        let mut actual = Vec::new();
        while let Some(row) = iter.next().await.unwrap() {
            actual.push(row);
        }
        assert_eq!(actual, rows);

        let mut oversized = WalSstIterator::new(table, store, WalSstIteratorOptions::default())
            .await
            .unwrap();
        oversized
            .prefetch_initial_blocks(first_block_size - 1)
            .await
            .unwrap();
        assert!(
            oversized.prefetched.is_none(),
            "an oversized block must not exceed the speculative budget"
        );
        assert_eq!(oversized.next().await.unwrap(), Some(rows[0].clone()));
    }
}
