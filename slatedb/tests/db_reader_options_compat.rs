use object_store::memory::InMemory;
use object_store::ObjectStore;
use slatedb::config::{DbReaderOptions, MetricLevel, ObjectStoreCacheOptions};
use slatedb::{Db, DbReader};
use std::sync::Arc;
use std::time::Duration;

#[test]
fn downstream_exhaustive_db_reader_options_literal_remains_source_compatible() {
    let options = DbReaderOptions {
        manifest_poll_interval: Duration::from_secs(10),
        checkpoint_lifetime: Duration::from_secs(10 * 60),
        max_memtable_bytes: 64 * 1024 * 1024,
        object_store_cache_options: ObjectStoreCacheOptions::default(),
        skip_wal_replay: false,
        metric_level: Some(MetricLevel::Info),
        object_store_max_retries: None,
    };

    assert_eq!(options.max_memtable_bytes, 64 * 1024 * 1024);
}

#[tokio::test]
async fn downstream_can_configure_reader_wal_replay_concurrency() {
    let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let db = Db::open("reader-options-compat", Arc::clone(&object_store))
        .await
        .unwrap();
    db.close().await.unwrap();

    let result = DbReader::builder("reader-options-compat", object_store)
        .with_wal_replay_concurrency(0)
        .build()
        .await;

    let error = match result {
        Ok(_) => panic!("zero replay concurrency should be rejected"),
        Err(error) => error,
    };
    assert_eq!(
        error.to_string(),
        "Invalid error: invalid sst batch size. size=`0`"
    );
}
