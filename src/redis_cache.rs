use crate::{Cache, CacheEntry, SemanticMatch};
use async_trait::async_trait;
use redis::{aio::ConnectionManager, AsyncCommands};

/// Durable exact-response cache backed by Redis.
///
/// Keys are namespaced and versioned so incompatible request canonicalization
/// changes do not collide with prior entries. This tier intentionally does not
/// answer semantic lookups; pair it with a pgvector-backed cache using
/// [`crate::TieredCache`].
pub struct RedisExactCache {
    connection: ConnectionManager,
    namespace: String,
    ttl_seconds: u64,
}

impl RedisExactCache {
    pub async fn connect(
        redis_url: &str,
        namespace: impl Into<String>,
        ttl_seconds: u64,
    ) -> Result<Self, String> {
        if ttl_seconds == 0 {
            return Err("Redis cache TTL must be greater than zero".into());
        }
        let client = redis::Client::open(redis_url).map_err(|error| error.to_string())?;
        let connection = ConnectionManager::new(client)
            .await
            .map_err(|error| error.to_string())?;
        Ok(Self {
            connection,
            namespace: namespace.into(),
            ttl_seconds,
        })
    }

    fn redis_key(&self, key: &str) -> String {
        format!("{}:response-cache:v1:{key}", self.namespace)
    }
}

#[async_trait]
impl Cache for RedisExactCache {
    async fn get_exact(&self, key: &str) -> Result<Option<CacheEntry>, String> {
        let mut connection = self.connection.clone();
        let payload: Option<String> = connection
            .get(self.redis_key(key))
            .await
            .map_err(|error| error.to_string())?;
        payload
            .map(|value| {
                serde_json::from_str(&value)
                    .map_err(|error| format!("invalid Redis cache entry: {error}"))
            })
            .transpose()
    }

    async fn put_exact(&self, key: String, entry: CacheEntry) -> Result<(), String> {
        let payload = serde_json::to_string(&entry).map_err(|error| error.to_string())?;
        let mut connection = self.connection.clone();
        let _: () = connection
            .set_ex(self.redis_key(&key), payload, self.ttl_seconds)
            .await
            .map_err(|error| error.to_string())?;
        Ok(())
    }

    async fn find_similar(&self, _: &[f32], _: f32) -> Result<Option<SemanticMatch>, String> {
        Ok(None)
    }
}
