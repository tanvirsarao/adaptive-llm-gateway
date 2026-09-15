use crate::Completion;
use async_trait::async_trait;
use std::{collections::HashMap, sync::RwLock};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CacheEntry {
    pub completion: Completion,
    pub embedding: Option<Vec<f32>>,
}
#[derive(Clone, Debug)]
pub struct SemanticMatch {
    pub entry: CacheEntry,
    pub similarity: f32,
}

#[async_trait]
pub trait Cache: Send + Sync {
    async fn get_exact(&self, key: &str) -> Result<Option<CacheEntry>, String>;
    async fn put_exact(&self, key: String, entry: CacheEntry) -> Result<(), String>;
    async fn find_similar(
        &self,
        embedding: &[f32],
        minimum_similarity: f32,
    ) -> Result<Option<SemanticMatch>, String>;
}

/// Combines independent persistent cache tiers behind the gateway's single
/// cache contract. For example: Redis for exact keys and pgvector for semantic
/// matches. Writes fan out so either tier can answer a later request.
pub struct TieredCache {
    exact: std::sync::Arc<dyn Cache>,
    semantic: std::sync::Arc<dyn Cache>,
}

impl TieredCache {
    pub fn new(exact: std::sync::Arc<dyn Cache>, semantic: std::sync::Arc<dyn Cache>) -> Self {
        Self { exact, semantic }
    }
}

#[async_trait]
impl Cache for TieredCache {
    async fn get_exact(&self, key: &str) -> Result<Option<CacheEntry>, String> {
        self.exact.get_exact(key).await
    }

    async fn put_exact(&self, key: String, entry: CacheEntry) -> Result<(), String> {
        self.exact.put_exact(key.clone(), entry.clone()).await?;
        self.semantic.put_exact(key, entry).await
    }

    async fn find_similar(
        &self,
        embedding: &[f32],
        minimum_similarity: f32,
    ) -> Result<Option<SemanticMatch>, String> {
        self.semantic
            .find_similar(embedding, minimum_similarity)
            .await
    }
}

/// Development cache. In production implement [`Cache`] with Redis for exact keys
/// and pgvector's cosine-distance operator for semantic lookups.
#[derive(Default)]
pub struct InMemoryCache {
    entries: RwLock<HashMap<String, CacheEntry>>,
}

#[async_trait]
impl Cache for InMemoryCache {
    async fn get_exact(&self, key: &str) -> Result<Option<CacheEntry>, String> {
        Ok(self
            .entries
            .read()
            .map_err(|_| "cache lock poisoned")?
            .get(key)
            .cloned())
    }
    async fn put_exact(&self, key: String, entry: CacheEntry) -> Result<(), String> {
        self.entries
            .write()
            .map_err(|_| "cache lock poisoned")?
            .insert(key, entry);
        Ok(())
    }
    async fn find_similar(
        &self,
        needle: &[f32],
        min: f32,
    ) -> Result<Option<SemanticMatch>, String> {
        let entries = self.entries.read().map_err(|_| "cache lock poisoned")?;
        Ok(entries
            .values()
            .filter_map(|entry| {
                entry
                    .embedding
                    .as_ref()
                    .map(|haystack| (entry, cosine(needle, haystack)))
            })
            .filter(|(_, score)| *score >= min)
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(entry, similarity)| SemanticMatch {
                entry: entry.clone(),
                similarity,
            }))
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return -1.0;
    }
    let (dot, aa, bb) = a.iter().zip(b).fold((0.0, 0.0, 0.0), |(d, x, y), (p, q)| {
        (d + p * q, x + p * p, y + q * q)
    });
    if aa == 0.0 || bb == 0.0 {
        -1.0
    } else {
        dot / (aa.sqrt() * bb.sqrt())
    }
}
