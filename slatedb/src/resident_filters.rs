//! Decoded SST filters kept on the SST's handle, so a read probes them without
//! a cache lookup.
//!
//! A point read probes the filter of every SST that may hold its key: each L0
//! SST and one SST per sorted run. Through the block cache each probe costs a
//! cache key, a hash, a shard lock and a reference count before the bloom
//! check itself, which is a hash and a few bit tests. A busy writer makes
//! millions of these probes, against the same few dozen filters.
//!
//! A [`ResidentFilterBudget`] lets a `Db` or `DbReader` keep each SST's
//! decoded filters on the SST's handle the first time they are read. Clones of
//! a handle share one slot, so every later read of that SST finds the filters
//! there. The slot returns its bytes to the budget when the last clone of the
//! handle is dropped, which happens once no manifest state refers to the SST.
//!
//! The budget bounds what is kept. One budget can be shared by every database
//! in a process, so a process with many databases has one cap rather than one
//! per database. When the budget is spent, reads go through the block cache as
//! they would without one.

use std::fmt::{Debug, Formatter};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use crate::filter_policy::NamedFilter;

/// The bytes of decoded filters the databases handed this budget may keep
/// resident on their SST handles.
pub struct ResidentFilterBudget {
    capacity_bytes: u64,
    used_bytes: AtomicU64,
}

impl ResidentFilterBudget {
    /// A budget of `capacity_bytes`. Zero keeps nothing resident.
    pub fn new(capacity_bytes: u64) -> Arc<Self> {
        Arc::new(Self {
            capacity_bytes,
            used_bytes: AtomicU64::new(0),
        })
    }

    /// The budget's capacity.
    pub fn capacity_bytes(&self) -> u64 {
        self.capacity_bytes
    }

    /// Bytes of filters resident now.
    pub fn used_bytes(&self) -> u64 {
        self.used_bytes.load(Ordering::Relaxed)
    }

    /// Takes `bytes` from the budget, or nothing if they do not fit.
    fn charge(self: &Arc<Self>, bytes: u64) -> Option<Charge> {
        let mut used = self.used_bytes.load(Ordering::Relaxed);
        loop {
            let total = used
                .checked_add(bytes)
                .filter(|total| *total <= self.capacity_bytes)?;
            match self.used_bytes.compare_exchange_weak(
                used,
                total,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(current) => used = current,
            }
        }
        Some(Charge {
            budget: Arc::clone(self),
            bytes,
        })
    }
}

impl Debug for ResidentFilterBudget {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResidentFilterBudget")
            .field("capacity_bytes", &self.capacity_bytes)
            .field("used_bytes", &self.used_bytes())
            .finish()
    }
}

/// Bytes taken from a budget, given back when dropped.
struct Charge {
    budget: Arc<ResidentFilterBudget>,
    bytes: u64,
}

impl Drop for Charge {
    fn drop(&mut self) {
        self.budget
            .used_bytes
            .fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

struct Resident {
    filters: Arc<[NamedFilter]>,
    _charge: Charge,
}

/// The resident filters of one SST, shared by every clone of its handle.
/// Empty until a read under a budget fills it.
#[derive(Default)]
pub(crate) struct ResidentSlot(OnceLock<Resident>);

impl ResidentSlot {
    /// The SST's filters, if they are resident.
    pub(crate) fn get(&self) -> Option<&Arc<[NamedFilter]>> {
        self.0.get().map(|resident| &resident.filters)
    }

    /// Keeps `filters` resident if `budget` has `bytes` for them. A slot that
    /// is already filled keeps its filters, and the new charge is returned.
    pub(crate) fn fill(
        &self,
        filters: &Arc<[NamedFilter]>,
        budget: &Arc<ResidentFilterBudget>,
        bytes: u64,
    ) {
        if self.0.get().is_some() {
            return;
        }
        if let Some(charge) = budget.charge(bytes) {
            // A racing fill wins; this one's charge drops with the value.
            let _ = self.0.set(Resident {
                filters: Arc::clone(filters),
                _charge: charge,
            });
        }
    }
}

/// A handle's identity is the SST it names, not whether its filters are
/// resident yet.
impl PartialEq for ResidentSlot {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Debug for ResidentSlot {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.get().is_some() {
            "ResidentSlot(resident)"
        } else {
            "ResidentSlot(empty)"
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filters() -> Arc<[NamedFilter]> {
        Arc::from([])
    }

    #[test]
    fn a_fill_within_the_budget_is_kept_and_released_on_drop() {
        let budget = ResidentFilterBudget::new(100);
        let slot = ResidentSlot::default();
        slot.fill(&filters(), &budget, 60);
        assert!(slot.get().is_some());
        assert_eq!(budget.used_bytes(), 60);
        drop(slot);
        assert_eq!(budget.used_bytes(), 0);
    }

    #[test]
    fn a_fill_over_the_budget_keeps_nothing() {
        let budget = ResidentFilterBudget::new(100);
        let first = ResidentSlot::default();
        first.fill(&filters(), &budget, 60);
        let second = ResidentSlot::default();
        second.fill(&filters(), &budget, 60);
        assert!(second.get().is_none());
        assert_eq!(budget.used_bytes(), 60);
    }

    #[test]
    fn a_filled_slot_is_not_charged_twice() {
        let budget = ResidentFilterBudget::new(100);
        let slot = ResidentSlot::default();
        slot.fill(&filters(), &budget, 40);
        slot.fill(&filters(), &budget, 40);
        assert_eq!(budget.used_bytes(), 40);
    }

    #[test]
    fn a_zero_budget_keeps_nothing() {
        let budget = ResidentFilterBudget::new(0);
        let slot = ResidentSlot::default();
        slot.fill(&filters(), &budget, 1);
        assert!(slot.get().is_none());
    }

    async fn db_with_flushed_sst(budget: Arc<ResidentFilterBudget>) -> crate::Db {
        use crate::config::{FlushOptions, FlushType, Settings};
        let db = crate::Db::builder(
            "resident-filters",
            Arc::new(object_store::memory::InMemory::new()),
        )
        .with_settings(Settings {
            compactor_options: None,
            ..Default::default()
        })
        .with_resident_filter_budget(budget)
        .build()
        .await
        .unwrap();
        // Enough keys that the SST is built with a filter.
        let mut batch = crate::WriteBatch::new();
        for i in 0..2_000_u32 {
            batch.put(format!("key-{i:05}"), format!("value-{i}"));
        }
        db.write(batch).await.unwrap();
        db.flush_with_options(FlushOptions {
            flush_type: FlushType::MemTable,
        })
        .await
        .unwrap();
        db
    }

    #[tokio::test]
    async fn a_read_keeps_the_sst_filters_resident_and_close_releases_them() {
        let budget = ResidentFilterBudget::new(1 << 20);
        let db = db_with_flushed_sst(Arc::clone(&budget)).await;
        assert_eq!(budget.used_bytes(), 0, "nothing is resident before a read");

        assert_eq!(
            db.get(b"key-00042").await.unwrap().as_deref(),
            Some(b"value-42".as_slice())
        );
        let resident = budget.used_bytes();
        assert!(resident > 0, "the read keeps the SST filters resident");

        // Later reads use the resident filters and still answer correctly,
        // both for present keys and for keys the filter rules out.
        for i in (0..2_000_u32).step_by(97) {
            let value = db.get(format!("key-{i:05}")).await.unwrap();
            assert_eq!(value.as_deref(), Some(format!("value-{i}").as_bytes()));
        }
        for i in 2_000..2_050_u32 {
            assert!(db.get(format!("key-{i:05}")).await.unwrap().is_none());
        }
        assert_eq!(budget.used_bytes(), resident, "each SST is charged once");

        db.close().await.unwrap();
        drop(db);
        assert_eq!(
            budget.used_bytes(),
            0,
            "closing the database releases its filters"
        );
    }

    #[tokio::test]
    async fn a_spent_budget_reads_through_the_cache() {
        let budget = ResidentFilterBudget::new(1);
        let db = db_with_flushed_sst(Arc::clone(&budget)).await;
        assert_eq!(
            db.get(b"key-00042").await.unwrap().as_deref(),
            Some(b"value-42".as_slice())
        );
        assert!(db.get(b"key-99999").await.unwrap().is_none());
        assert_eq!(
            budget.used_bytes(),
            0,
            "a filter larger than the budget is not kept"
        );
        db.close().await.unwrap();
    }
}
