use std::{collections::HashMap, future::Future, hash::Hash, sync::Arc};

use futures::future::{BoxFuture, FutureExt, Shared};
use parking_lot::Mutex as SyncMutex;

/// The leader's func returned an error. The leader gets the error itself;
/// every other caller of the flight retries with its own func.
#[derive(Clone, Copy, Debug)]
struct FlightFailed;

/// One in-flight call. Every caller of the key polls the same shared future,
/// so the call makes progress while any caller is polled. A leader that stops
/// polling without dropping its call cannot block the others.
type Flight<T> = Shared<BoxFuture<'static, Result<T, FlightFailed>>>;

/// A key's flight and the number of callers that hold it.
struct Entry<T: Clone> {
    flight: Flight<T>,
    callers: usize,
}

type InFlightMap<K, T> = Arc<SyncMutex<HashMap<K, Entry<T>>>>;

// FlightGuard counts its caller in the key's entry while the caller waits. The
// last caller to leave removes the entry, so a later call starts a fresh flight.
// Removing it drops the map's handle on the shared future, and once every
// caller's handle is gone too, a flight that is not done is cancelled.
struct FlightGuard<'a, K: Hash + Eq, T: Clone> {
    flight: Flight<T>,
    key: &'a K,
    in_flight: &'a InFlightMap<K, T>,
}

impl<'a, K, T> Drop for FlightGuard<'a, K, T>
where
    K: Hash + Eq,
    T: Clone,
{
    fn drop(&mut self) {
        let mut in_flight = self.in_flight.lock();
        let Some(entry) = in_flight.get_mut(self.key) else {
            return;
        };
        // The entry may already belong to a newer flight of the same key.
        if !entry.flight.ptr_eq(&self.flight) {
            return;
        }
        entry.callers -= 1;
        if entry.callers == 0 {
            in_flight.remove(self.key);
        }
    }
}

/// SingleFlight deduplicates concurrent calls for the same key, ensuring only one
/// execution is in-flight at a time while sharing the result with all waiters.
#[derive(Clone)]
pub struct SingleFlight<K, T: Clone> {
    in_flight: InFlightMap<K, T>,
}

impl<K, T: Clone> std::fmt::Debug for SingleFlight<K, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SingleFlight")
            .field("in_flight", &self.in_flight.lock().len())
            .finish()
    }
}

impl<K, T: Clone> Default for SingleFlight<K, T> {
    fn default() -> Self {
        Self {
            in_flight: Default::default(),
        }
    }
}

impl<K, T> SingleFlight<K, T>
where
    K: Hash + Eq + Clone,
    T: Clone + Send + Sync + 'static,
{
    /// Create a new SingleFlight group.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Execute and return the value for a given function, making sure that only one
    /// operation is in-flight at a given moment. If a duplicate call comes in, that caller will
    /// wait until the original call completes and return the same value.
    ///
    /// Every caller of a key polls the same shared future, so the call does not depend on the
    /// caller that started it. If that caller stops polling or is dropped, the others still
    /// finish it. If the call fails, its caller gets the error and the others retry with their
    /// own `func`.
    pub async fn call<F, Fut, E>(&self, key: K, func: F) -> Result<T, E>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<T, E>> + Send + 'static,
        E: Send + 'static,
    {
        let mut func = Some(func);
        loop {
            // Join the key's flight, or start one, while briefly holding the lock.
            let (guard, error) = {
                let mut in_flight = self.in_flight.lock();
                let (flight, error) = match in_flight.get_mut(&key) {
                    Some(entry) => {
                        entry.callers += 1;
                        (entry.flight.clone(), None)
                    }
                    None => {
                        // Only the caller whose func ran leaves the loop on a failure, so a
                        // caller that starts a flight still holds its func.
                        let func = func.take().expect("a caller starts at most one flight");
                        let error = Arc::new(SyncMutex::new(None));
                        let slot = Arc::clone(&error);
                        let call = func();
                        let flight = async move {
                            call.await.map_err(|err| {
                                *slot.lock() = Some(err);
                                FlightFailed
                            })
                        }
                        .boxed()
                        .shared();
                        in_flight.insert(
                            key.clone(),
                            Entry {
                                flight: flight.clone(),
                                callers: 1,
                            },
                        );
                        (flight, Some(error))
                    }
                };
                let guard = FlightGuard {
                    flight,
                    key: &key,
                    in_flight: &self.in_flight,
                };
                (guard, error)
            };
            let outcome = guard.flight.clone().await;
            if outcome.is_err() {
                // A failed flight is never joined again, even while other
                // callers still hold it: a retry must start a fresh one.
                let mut in_flight = self.in_flight.lock();
                if in_flight
                    .get(&key)
                    .is_some_and(|entry| entry.flight.ptr_eq(&guard.flight))
                {
                    in_flight.remove(&key);
                }
            }
            drop(guard);
            match (outcome, error) {
                (Ok(value), _) => return Ok(value),
                (Err(FlightFailed), Some(error)) => {
                    let err = error.lock().take();
                    return Err(err.expect("a failed flight stores its error"));
                }
                // Another caller's func failed: run our own.
                (Err(FlightFailed), None) => continue,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{
            AtomicUsize,
            Ordering::{AcqRel, Acquire},
        },
        Arc,
    };

    use tokio::sync::Notify;

    use super::*;

    #[tokio::test]
    async fn direct_call() {
        let group = SingleFlight::new();
        let result = group
            .call("key", || async { Ok::<_, ()>("Result".to_string()) })
            .await;
        assert_eq!(result, Ok("Result".to_string()));
    }

    #[tokio::test]
    async fn parallel_call() {
        let call_counter = Arc::new(AtomicUsize::default());
        let gate = Arc::new(Notify::new());

        let group = SingleFlight::<&str, String>::new();
        let mut handles = Vec::new();
        for _ in 0..10 {
            let g = group.clone();
            let counter = call_counter.clone();
            let gate = gate.clone();
            handles.push(tokio::spawn(async move {
                g.call("key", || async move {
                    gate.notified().await;
                    counter.fetch_add(1, AcqRel);
                    Ok::<_, ()>("Result".to_string())
                })
                .await
            }));
        }

        // Let all tasks register in the map.
        tokio::task::yield_now().await;

        // Release the single winner.
        gate.notify_one();

        for handle in handles {
            assert_eq!(
                handle.await.expect("task panicked"),
                Ok("Result".to_string())
            );
        }
        assert_eq!(
            call_counter.load(Acquire),
            1,
            "future should only be executed once"
        );
    }

    #[tokio::test]
    async fn different_keys_are_independent() {
        let counter_a = Arc::new(AtomicUsize::default());
        let counter_b = Arc::new(AtomicUsize::default());
        let gate_a = Arc::new(Notify::new());
        let gate_b = Arc::new(Notify::new());

        let group = SingleFlight::<&'static str, String>::new();

        let g = group.clone();
        let ca = counter_a.clone();
        let ga = gate_a.clone();
        let handle_a = tokio::spawn(async move {
            g.call("key_a", || async move {
                ga.notified().await;
                ca.fetch_add(1, AcqRel);
                Ok::<_, ()>("A".to_string())
            })
            .await
        });

        let g = group.clone();
        let cb = counter_b.clone();
        let gb = gate_b.clone();
        let handle_b = tokio::spawn(async move {
            g.call("key_b", || async move {
                gb.notified().await;
                cb.fetch_add(1, AcqRel);
                Ok::<_, ()>("B".to_string())
            })
            .await
        });

        // Let both tasks register.
        tokio::task::yield_now().await;

        // Release both independently.
        gate_a.notify_one();
        gate_b.notify_one();

        assert_eq!(handle_a.await.expect("task panicked"), Ok("A".to_string()));
        assert_eq!(handle_b.await.expect("task panicked"), Ok("B".to_string()));
        assert_eq!(counter_a.load(Acquire), 1);
        assert_eq!(counter_b.load(Acquire), 1);
    }

    #[tokio::test]
    async fn sequential_calls_run_fresh() {
        // After a call completes, a new call to the same key should execute its own func.
        let call_counter = Arc::new(AtomicUsize::default());

        let group = SingleFlight::new();

        let counter = call_counter.clone();
        let result = group
            .call("key", || async move {
                counter.fetch_add(1, AcqRel);
                Ok::<_, ()>("first".to_string())
            })
            .await;
        assert_eq!(result, Ok("first".to_string()));

        let counter = call_counter.clone();
        let result = group
            .call("key", || async move {
                counter.fetch_add(1, AcqRel);
                Ok::<_, ()>("second".to_string())
            })
            .await;
        assert_eq!(result, Ok("second".to_string()));

        assert_eq!(
            call_counter.load(Acquire),
            2,
            "each sequential call should execute independently"
        );
    }

    #[tokio::test]
    async fn error_propagates_and_next_caller_retries() {
        let group = SingleFlight::<&str, String>::new();

        // First call fails.
        let result: Result<String, &str> = group.call("key", || async { Err("oops") }).await;
        assert_eq!(result, Err("oops"));

        // Next call should run its own func (not get the cached error).
        let result = group
            .call("key", || async { Ok::<_, &str>("recovered".to_string()) })
            .await;
        assert_eq!(result, Ok("recovered".to_string()));
    }

    #[tokio::test]
    async fn error_with_concurrent_waiters() {
        // When the initializer fails, a concurrent waiter gets to retry with its own func.
        let group = SingleFlight::<&str, String>::new();
        let call_counter = Arc::new(AtomicUsize::default());
        let gate = Arc::new(Notify::new());

        let counter = call_counter.clone();
        let g = gate.clone();
        let fut_1 = group.call("key", || async move {
            g.notified().await;
            counter.fetch_add(1, AcqRel);
            Err::<String, _>("fail")
        });
        let counter = call_counter.clone();
        let fut_2 = group.call("key", || async move {
            counter.fetch_add(1, AcqRel);
            Ok::<_, &str>("recovered".to_string())
        });

        // Release the first caller so it fails.
        gate.notify_one();

        let (r1, r2) = tokio::join!(fut_1, fut_2);
        assert_eq!(r1, Err("fail"));
        assert_eq!(r2, Ok("recovered".to_string()));
        assert_eq!(
            call_counter.load(Acquire),
            2,
            "both funcs should have been called since the first failed"
        );
    }

    #[tokio::test]
    async fn call_with_custom_key() {
        #[derive(Clone, PartialEq, Eq, Hash)]
        struct K(i32);
        let group = SingleFlight::new();
        let result = group
            .call(K(1), || async { Ok::<_, ()>("Result".to_string()) })
            .await;
        assert_eq!(result, Ok("Result".to_string()));
    }

    #[tokio::test]
    async fn late_joiner_shares_result() {
        let group = SingleFlight::<String, String>::new();
        let gate = Arc::new(Notify::new());

        // Spawn early so it registers in the map while it's blocked on the gate.
        let g = group.clone();
        let gate_clone = gate.clone();
        let early_handle = tokio::spawn(async move {
            g.call("key".to_string(), || async move {
                gate_clone.notified().await;
                Ok::<_, ()>("Result".to_string())
            })
            .await
        });

        // Yield so the spawned task registers in the map.
        tokio::task::yield_now().await;

        // Late caller finds the existing entry — its func should never run.
        let late_fut = group.call("key".to_string(), || async { panic!("unexpected") });

        // Release the gate so the first caller completes.
        gate.notify_one();

        let (early_result, late_result) = tokio::join!(early_handle, late_fut);
        assert_eq!(
            early_result.expect("task panicked"),
            Ok("Result".to_string())
        );
        assert_eq!(late_result, Ok::<_, ()>("Result".to_string()));
    }

    #[tokio::test]
    async fn cancel_allows_next_caller_to_proceed() {
        let group = SingleFlight::new();

        // Start a call that will never complete on its own.
        let handle = tokio::spawn({
            let g = group.clone();
            async move {
                g.call("key".to_string(), || {
                    std::future::pending::<Result<String, ()>>()
                })
                .await
            }
        });

        // Let it register.
        tokio::task::yield_now().await;

        // Cancel it.
        handle.abort();
        let _ = handle.await;

        // The next caller should run fresh since the previous was cancelled.
        let result = group
            .call("key".to_string(), || async {
                Ok::<_, ()>("fresh".to_string())
            })
            .await;
        assert_eq!(result, Ok("fresh".to_string()));
    }

    #[tokio::test]
    async fn cancel_with_concurrent_waiter() {
        // If the initializer is cancelled while another caller waits, the waiter finishes the same
        // call. It does not run its own func.
        let group = SingleFlight::<String, String>::new();
        let gate = Arc::new(Notify::new());

        let g = group.clone();
        let initializer_gate = gate.clone();
        let initializer = tokio::spawn(async move {
            g.call("key".to_string(), || async move {
                initializer_gate.notified().await;
                Ok::<_, ()>("shared".to_string())
            })
            .await
        });

        // Let the initializer register.
        tokio::task::yield_now().await;

        // Start a concurrent waiter on the same key.
        let g = group.clone();
        let waiter = tokio::spawn(async move {
            g.call("key".to_string(), || async {
                panic!("the waiter must finish the initializer's call");
                #[allow(unreachable_code)]
                Ok::<_, ()>(String::new())
            })
            .await
        });

        // Let the waiter register and start waiting.
        tokio::task::yield_now().await;

        // Cancel the initializer, then let its call finish.
        initializer.abort();
        let _ = initializer.await;
        gate.notify_one();

        let result = waiter.await.expect("task panicked");
        assert_eq!(result, Ok("shared".to_string()));
    }

    // A caller can poll its call once, so it starts the flight, and then
    // stop polling it without dropping it: an SST iterator keeps a block
    // fetch in its queue when its own caller cancels `next`. The flight must
    // still finish for every other caller of that key. On turbolay-prod
    // (2026-11-18) a parked fetch held block reads of one SST part for
    // 40 minutes while its leader's own S3 get never ran.
    #[tokio::test]
    async fn parked_leader_does_not_block_waiters() {
        let group = SingleFlight::<&str, String>::new();
        let gate = Arc::new(Notify::new());

        let leader_gate = gate.clone();
        let mut leader = Box::pin(group.call("key", move || async move {
            leader_gate.notified().await;
            Ok::<_, ()>("value".to_string())
        }));
        // Poll the leader until its func is waiting on the gate, then park it.
        tokio::select! {
            biased;
            _ = &mut leader => panic!("the leader cannot finish before the gate opens"),
            _ = tokio::task::yield_now() => {}
        }
        gate.notify_one();

        let waiter = group.call("key", || async {
            panic!("the waiter must share the leader's flight")
        });
        let result = tokio::time::timeout(std::time::Duration::from_secs(5), waiter)
            .await
            .expect("a parked leader must not block its waiters");
        assert_eq!(result, Ok::<_, ()>("value".to_string()));
        drop(leader);
    }

    #[tokio::test]
    async fn concurrent_callers_share_result() {
        // Two concurrent callers — second should not run its func.
        let group = SingleFlight::<String, String>::new();
        let gate = Arc::new(Notify::new());

        // Spawn the first caller so it registers and blocks on the gate.
        let g = group.clone();
        let gate_clone = gate.clone();
        let handle = tokio::spawn(async move {
            g.call("key".to_string(), || async move {
                gate_clone.notified().await;
                Ok::<_, ()>("Result1".to_string())
            })
            .await
        });

        // Let the spawned task register in the map.
        tokio::task::yield_now().await;

        // Second caller finds the existing entry — its func should never run.
        let fut_2 = group.call("key".to_string(), || async { panic!("should not execute") });

        // Release the gate so the first caller completes.
        gate.notify_one();

        let (v1, v2) = tokio::join!(handle, fut_2);
        assert_eq!(v1.expect("task panicked"), Ok("Result1".to_string()));
        assert_eq!(v2, Ok::<_, ()>("Result1".to_string()));
    }

    #[tokio::test]
    async fn map_is_cleaned_up_after_completion() {
        let group = SingleFlight::new();

        // Run a call to completion.
        let _ = group
            .call("key", || async { Ok::<_, ()>("done".to_string()) })
            .await;

        // The internal map should be empty.
        let in_flight = group.in_flight.lock();
        assert!(
            in_flight.is_empty(),
            "map should be empty after all callers complete"
        );
    }

    #[tokio::test]
    async fn map_is_cleaned_up_after_cancel() {
        let group = SingleFlight::new();

        // Start a call that will never complete on its own.
        let handle = tokio::spawn({
            let g = group.clone();
            async move {
                g.call("key".to_string(), || {
                    std::future::pending::<Result<String, ()>>()
                })
                .await
            }
        });

        // Let it register.
        tokio::task::yield_now().await;

        // Cancel it.
        handle.abort();
        let _ = handle.await;

        // The internal map should be empty after cancellation.
        let in_flight = group.in_flight.lock();
        assert!(
            in_flight.is_empty(),
            "map should be empty after cancellation"
        );
    }
}
