use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub prompt: String,
    /// Instructions that change the meaning of a prompt and must be part of
    /// the cache fingerprint and provider request.
    #[serde(default)]
    pub system: Option<String>,
    pub model: Option<String>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    /// Supplies a precomputed embedding when semantic caching is enabled.
    pub embedding: Option<Vec<f32>>,
    /// Isolation boundary for cache ownership (usually a tenant or user ID).
    #[serde(default = "default_cache_scope")]
    pub cache_scope: String,
    /// Declares whether an approximate answer may be reused and under what
    /// output contract. This is part of cache identity.
    #[serde(default)]
    pub cache_policy: CachePolicy,
}

fn default_cache_scope() -> String {
    "default".into()
}

impl CompletionRequest {
    pub fn new(prompt: String) -> Self {
        Self {
            prompt,
            system: None,
            model: None,
            max_tokens: None,
            temperature: None,
            embedding: None,
            cache_scope: default_cache_scope(),
            cache_policy: CachePolicy::default(),
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
    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }
    pub fn cache_scope(mut self, scope: impl Into<String>) -> Self {
        self.cache_scope = scope.into();
        self
    }
    pub fn cache_policy(mut self, policy: CachePolicy) -> Self {
        self.cache_policy = policy;
        self
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum SemanticReuse {
    /// An answer is never reused for an approximate request.
    Disabled,
    /// Safe for read-only, non-user-specific work such as summarization or
    /// classification. The caller owns the suitability decision.
    SafeReadOnly,
    /// Reserved for a future cheap verifier before serving an approximate hit.
    VerifyBeforeUse,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CachePolicy {
    /// Identifies the expected task/output contract, e.g. `summary:v1` or
    /// `sentiment-json:v2`. Different contracts never share semantic hits.
    pub compatibility_key: String,
    pub semantic_reuse: SemanticReuse,
}

impl Default for CachePolicy {
    fn default() -> Self {
        Self {
            compatibility_key: "text-generation:v1".into(),
            semantic_reuse: SemanticReuse::Disabled,
        }
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
    /// What this request cost at the selected provider, in USD.
    pub cost: CostSummary,
    /// Decision data for displaying or exporting the route the gateway took.
    pub route: RouteTrace,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CostSummary {
    pub input_usd: f64,
    pub output_usd: f64,
    /// Spend incurred by verifier/planner models during delegation.
    pub verification_usd: f64,
    pub total_usd: f64,
    /// The provider cost not paid because this response came from cache.
    pub avoided_usd: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RouteTrace {
    /// `miss`, `exact`, or `semantic`.
    pub cache: String,
    /// Cosine similarity for a semantic-cache hit.
    pub semantic_similarity: Option<f32>,
    /// Cost-ranked provider options considered for this request.
    pub candidates: Vec<RouteCandidate>,
    /// Providers called in order; a cached response has no attempts.
    pub attempts: Vec<ProviderAttempt>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RouteCandidate {
    pub provider: String,
    pub model: String,
    pub input_cost_per_1k_usd: f64,
    pub output_cost_per_1k_usd: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProviderAttempt {
    pub provider: String,
    pub outcome: String,
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
