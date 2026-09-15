use crate::{Cache, CacheEntry, SemanticMatch};
use async_trait::async_trait;
use std::sync::Arc;
use tokio_postgres::{Client, NoTls};

/// Durable semantic-response cache backed by PostgreSQL and pgvector.
///
/// A cache instance is deliberately bound to one tenant scope and one
/// compatibility key. This prevents a semantically similar answer from a
/// different tenant or task/output contract being reused accidentally.
pub struct PgVectorSemanticCache {
    client: Arc<Client>,
    tenant_scope: String,
    compatibility_key: String,
    ttl_seconds: i64,
}

impl PgVectorSemanticCache {
    /// SQL migration for a database role that is allowed to install pgvector.
    pub const SCHEMA_SQL: &'static str = r#"
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS llm_semantic_cache (
    id BIGSERIAL PRIMARY KEY,
    tenant_scope TEXT NOT NULL,
    compatibility_key TEXT NOT NULL,
    cache_key TEXT NOT NULL,
    embedding vector NOT NULL,
    entry JSONB NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (tenant_scope, compatibility_key, cache_key)
);
CREATE INDEX IF NOT EXISTS llm_semantic_cache_expiry_idx ON llm_semantic_cache (expires_at);
CREATE INDEX IF NOT EXISTS llm_semantic_cache_scope_idx ON llm_semantic_cache (tenant_scope, compatibility_key);
"#;

    pub async fn connect(
        database_url: &str,
        tenant_scope: impl Into<String>,
        compatibility_key: impl Into<String>,
        ttl_seconds: i64,
    ) -> Result<Self, String> {
        if ttl_seconds <= 0 {
            return Err("pgvector cache TTL must be greater than zero".into());
        }
        let (client, connection) = tokio_postgres::connect(database_url, NoTls)
            .await
            .map_err(|error| error.to_string())?;
        tokio::spawn(async move {
            if let Err(error) = connection.await {
                tracing::warn!(%error, "pgvector cache connection closed");
            }
        });
        Ok(Self {
            client: Arc::new(client),
            tenant_scope: tenant_scope.into(),
            compatibility_key: compatibility_key.into(),
            ttl_seconds,
        })
    }

    /// Create the pgvector extension, table, and indexes. Prefer applying
    /// [`Self::SCHEMA_SQL`] through your normal migration system in production.
    pub async fn initialize(&self) -> Result<(), String> {
        self.client
            .batch_execute(Self::SCHEMA_SQL)
            .await
            .map_err(|error| error.to_string())
    }
}

#[async_trait]
impl Cache for PgVectorSemanticCache {
    async fn get_exact(&self, _: &str) -> Result<Option<CacheEntry>, String> {
        Ok(None)
    }

    async fn put_exact(&self, key: String, entry: CacheEntry) -> Result<(), String> {
        let Some(embedding) = &entry.embedding else {
            return Ok(());
        };
        let entry = serde_json::to_string(&entry).map_err(|error| error.to_string())?;
        let embedding = vector_literal(embedding);
        self.client.execute(
            "INSERT INTO llm_semantic_cache (tenant_scope, compatibility_key, cache_key, embedding, entry, expires_at)
             VALUES ($1, $2, $3, $4::vector, $5::jsonb, NOW() + ($6 * INTERVAL '1 second'))
             ON CONFLICT (tenant_scope, compatibility_key, cache_key) DO UPDATE SET
               embedding = EXCLUDED.embedding, entry = EXCLUDED.entry, expires_at = EXCLUDED.expires_at, created_at = NOW()",
            &[&self.tenant_scope, &self.compatibility_key, &key, &embedding, &entry, &self.ttl_seconds],
        ).await.map_err(|error| error.to_string())?;
        Ok(())
    }

    async fn find_similar(
        &self,
        embedding: &[f32],
        minimum_similarity: f32,
    ) -> Result<Option<SemanticMatch>, String> {
        let embedding = vector_literal(embedding);
        let row = self
            .client
            .query_opt(
                "SELECT entry::text, 1 - (embedding <=> $3::vector) AS similarity
             FROM llm_semantic_cache
             WHERE tenant_scope = $1 AND compatibility_key = $2 AND expires_at > NOW()
               AND 1 - (embedding <=> $3::vector) >= $4
             ORDER BY embedding <=> $3::vector ASC LIMIT 1",
                &[
                    &self.tenant_scope,
                    &self.compatibility_key,
                    &embedding,
                    &(minimum_similarity as f64),
                ],
            )
            .await
            .map_err(|error| error.to_string())?;
        row.map(|row| {
            let entry: String = row.try_get(0).map_err(|error| error.to_string())?;
            let similarity: f64 = row.try_get(1).map_err(|error| error.to_string())?;
            let entry = serde_json::from_str(&entry)
                .map_err(|error| format!("invalid pgvector cache entry: {error}"))?;
            Ok(SemanticMatch {
                entry,
                similarity: similarity as f32,
            })
        })
        .transpose()
    }
}

fn vector_literal(values: &[f32]) -> String {
    let values = values
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(",");
    format!("[{values}]")
}
