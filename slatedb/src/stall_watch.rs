//! Logs a read that stays pending too long, and names what it waits on.
//!
//! A scan open can park forever when a read below it never completes. Nothing
//! then shows which SST, cache key, or object part holds it. [`watch`] wraps one
//! wait. After [`FIRST_REPORT`] it logs a warning that names the wait. It logs
//! again every [`REPEAT_REPORT`], once more when the wait completes, and once
//! if the caller drops the wait first.
//!
//! The watch is off until a process calls [`enable`] with a clock. When it is
//! off, [`watch`] only awaits the future. Tests and deterministic simulation
//! therefore see no timers and no change in behavior.

use std::future::Future;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use chrono::{DateTime, Utc};
use log::warn;
use slatedb_common::clock::SystemClock;

/// How long a wait can stay pending before the first warning.
pub const FIRST_REPORT: Duration = Duration::from_secs(10);

/// How often a wait that stays pending is reported again.
pub const REPEAT_REPORT: Duration = Duration::from_secs(30);

static CLOCK: OnceLock<Arc<dyn SystemClock>> = OnceLock::new();

/// Turns the watch on for this process. Returns `false` if it was already on.
pub fn enable(clock: Arc<dyn SystemClock>) -> bool {
    CLOCK.set(clock).is_ok()
}

/// Awaits `future`. If the watch is on and the wait stays pending past
/// [`FIRST_REPORT`], logs `stage` and the text from `describe`.
pub(crate) async fn watch<F, D>(stage: &'static str, describe: D, future: F) -> F::Output
where
    F: Future,
    D: Fn() -> String,
{
    match CLOCK.get() {
        None => future.await,
        Some(clock) => {
            watch_with(
                clock.as_ref(),
                FIRST_REPORT,
                REPEAT_REPORT,
                stage,
                describe,
                future,
            )
            .await
        }
    }
}

async fn watch_with<F, D>(
    clock: &dyn SystemClock,
    first: Duration,
    repeat: Duration,
    stage: &'static str,
    describe: D,
    future: F,
) -> F::Output
where
    F: Future,
    D: Fn() -> String,
{
    let mut future = std::pin::pin!(future);
    let mut guard = DropReport {
        clock,
        started: clock.now(),
        first,
        stage,
        describe: &describe,
        armed: true,
    };
    let mut reported = false;
    let mut sleep = clock.sleep(first);
    loop {
        tokio::select! {
            biased;
            output = &mut future => {
                guard.armed = false;
                if reported {
                    warn!(
                        "read wait completed after a long wait [stage={}, waited_ms={}, what={}]",
                        stage,
                        guard.waited_ms(),
                        describe()
                    );
                }
                return output;
            }
            () = &mut sleep => {
                reported = true;
                warn!(
                    "read wait still pending [stage={}, waited_ms={}, what={}]",
                    stage,
                    guard.waited_ms(),
                    describe()
                );
                sleep = clock.sleep(repeat);
            }
        }
    }
}

/// Logs the wait if the caller drops it after the first report time.
struct DropReport<'a, D: Fn() -> String> {
    clock: &'a dyn SystemClock,
    started: DateTime<Utc>,
    first: Duration,
    stage: &'static str,
    describe: &'a D,
    armed: bool,
}

impl<D: Fn() -> String> DropReport<'_, D> {
    fn waited_ms(&self) -> i64 {
        (self.clock.now() - self.started).num_milliseconds()
    }
}

impl<D: Fn() -> String> Drop for DropReport<'_, D> {
    fn drop(&mut self) {
        let waited_ms = self.waited_ms();
        if self.armed && waited_ms >= self.first.as_millis() as i64 {
            warn!(
                "read wait dropped while pending [stage={}, waited_ms={}, what={}]",
                self.stage,
                waited_ms,
                (self.describe)()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use slatedb_common::clock::MockSystemClock;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const FIRST: Duration = Duration::from_secs(10);
    const REPEAT: Duration = Duration::from_secs(30);

    #[tokio::test]
    async fn a_ready_wait_returns_its_output_without_reading_the_description() {
        let clock = MockSystemClock::new();
        let described = AtomicUsize::new(0);
        let describe = || {
            described.fetch_add(1, Ordering::SeqCst);
            "sst".to_string()
        };
        let output = watch_with(&clock, FIRST, REPEAT, "ready", describe, async { 7 }).await;
        assert_eq!(output, 7);
        assert_eq!(described.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn a_pending_wait_is_described_when_the_clock_passes_the_first_report() {
        let clock = Arc::new(MockSystemClock::new());
        let described = Arc::new(AtomicUsize::new(0));
        let (release, released) = tokio::sync::oneshot::channel::<u8>();
        let task = {
            let clock = Arc::clone(&clock);
            let described = Arc::clone(&described);
            tokio::spawn(async move {
                let describe = || {
                    described.fetch_add(1, Ordering::SeqCst);
                    "sst 01ABC blocks 3..4".to_string()
                };
                watch_with(
                    clock.as_ref(),
                    FIRST,
                    REPEAT,
                    "sst_block_fetch",
                    describe,
                    async { released.await.expect("released") },
                )
                .await
            })
        };
        tokio::task::yield_now().await;
        assert_eq!(
            described.load(Ordering::SeqCst),
            0,
            "nothing is reported before the first report time"
        );
        clock.advance(FIRST).await;
        wait_for(|| described.load(Ordering::SeqCst) == 1).await;
        release.send(9).expect("the wait is still pending");
        assert_eq!(task.await.expect("task"), 9);
        assert_eq!(
            described.load(Ordering::SeqCst),
            2,
            "a wait reported as pending is reported again when it completes"
        );
    }

    #[tokio::test]
    async fn the_watch_is_a_plain_await_until_enabled() {
        // `enable` is process-wide and tests share the process, so this only
        // checks the path that runs when no clock is set.
        if CLOCK.get().is_none() {
            assert_eq!(watch("off", || unreachable!(), async { 3 }).await, 3);
        }
    }

    async fn wait_for(condition: impl Fn() -> bool) {
        for _ in 0..1_000 {
            if condition() {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!("condition did not become true");
    }
}
