use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub prompt: String,
    pub model: Option<String>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    /// Supplies a precomputed embedding when semantic caching is enabled.
    pub embedding: Option<Vec<f32>>,
}

impl CompletionRequest {
    pub fn new(prompt: String) -> Self {
        Self {
            prompt,
            model: None,
            max_tokens: None,
            temperature: None,
            embedding: None,
        }
    }

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }
    pub fn embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Completion {
    pub id: Uuid,
    pub text: String,
    pub model: String,
    pub provider: String,
    pub usage: Usage,
    pub cached: bool,
    pub cache_kind: Option<String>,
}

#[derive(Debug, Error)]
pub enum GatewayError {
    #[error("no provider is configured for model `{0}`")]
    NoProvider(String),
    #[error("all providers failed: {0}")]
    ProvidersFailed(String),
    #[error("cache failure: {0}")]
    Cache(String),
    #[error("invalid configuration: {0}")]
    Configuration(String),
}
