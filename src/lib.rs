//! A provider-agnostic LLM gateway you can embed directly in a Rust service.
//!
//! ```no_run
//! use std::sync::Arc;
//! use adaptive_llm_gateway::{Gateway, GatewayConfig, InMemoryCache, CompletionRequest};
//! # async fn example() -> Result<(), adaptive_llm_gateway::GatewayError> {
//! let gateway = Gateway::builder(GatewayConfig::default())
//!     .cache(Arc::new(InMemoryCache::default()))
//!     // .provider(Arc::new(OpenAiProvider::from_env()?))
//!     .build()?;
//! let response = gateway.complete(CompletionRequest::new("Summarize this document".into())).await?;
//! # Ok(()) }
//! ```

mod cache;
mod gateway;
#[cfg(feature = "openai-compatible")]
mod openai_compatible;
#[cfg(feature = "pgvector-cache")]
mod pgvector_cache;
mod provider;
#[cfg(feature = "redis-cache")]
mod redis_cache;
mod types;

pub use cache::{Cache, CacheEntry, InMemoryCache, SemanticMatch, TieredCache};
pub use gateway::{Gateway, GatewayBuilder, GatewayConfig};
#[cfg(feature = "openai-compatible")]
pub use openai_compatible::OpenAiCompatibleProvider;
#[cfg(feature = "pgvector-cache")]
pub use pgvector_cache::PgVectorSemanticCache;
pub use provider::{LlmProvider, ProviderCompletion, ProviderError};
#[cfg(feature = "redis-cache")]
pub use redis_cache::RedisExactCache;
pub use types::{
    Completion, CompletionRequest, CostSummary, GatewayError, ProviderAttempt, RouteCandidate,
    RouteTrace, Usage,
};
