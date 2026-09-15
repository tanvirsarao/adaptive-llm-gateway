use crate::{CompletionRequest, Usage};
use async_trait::async_trait;
use thiserror::Error;

#[derive(Clone, Debug)]
pub struct ProviderCompletion {
    pub text: String,
    pub model: String,
    pub usage: Usage,
}

#[derive(Debug, Error)]
#[error("{message}")]
pub struct ProviderError {
    pub message: String,
    pub retryable: bool,
}

#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Stable identifier used in logs, responses, and policy configuration.
    fn name(&self) -> &str;
    /// Models this provider can serve. Order is irrelevant.
    fn models(&self) -> &[String];
    /// Estimated USD per 1K input tokens. Used only for routing.
    fn input_cost_per_1k(&self, model: &str) -> Option<f64>;
    /// Estimated USD per 1K output tokens. Used for request-cost reporting.
    fn output_cost_per_1k(&self, _model: &str) -> Option<f64> {
        None
    }
    async fn complete(
        &self,
        request: &CompletionRequest,
    ) -> Result<ProviderCompletion, ProviderError>;
}
