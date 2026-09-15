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
mod provider;
mod types;
#[cfg(feature = "openai-compatible")]
mod openai_compatible;

pub use cache::{Cache, CacheEntry, InMemoryCache, SemanticMatch};
pub use gateway::{Gateway, GatewayBuilder, GatewayConfig};
pub use provider::{LlmProvider, ProviderCompletion, ProviderError};
pub use types::{Completion, CompletionRequest, GatewayError, Usage};
#[cfg(feature = "openai-compatible")]
pub use openai_compatible::OpenAiCompatibleProvider;
