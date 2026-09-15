use crate::{CompletionRequest, CostSummary, LlmProvider};
use async_trait::async_trait;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct Verification {
    pub approved: bool,
    pub provider: String,
    pub reason: String,
    pub cost: CostSummary,
}

#[async_trait]
pub trait CompletionVerifier: Send + Sync {
    async fn verify(&self, request: &CompletionRequest, answer: &str) -> Verification;
}

/// Uses any configured LLM as a strict APPROVE/REJECT judge. Pair it with
/// cheaper execution providers to make a selective inference cascade.
pub struct LlmJudgeVerifier {
    provider: Arc<dyn LlmProvider>,
    model: Option<String>,
}

impl LlmJudgeVerifier {
    pub fn new(provider: Arc<dyn LlmProvider>) -> Self {
        Self {
            provider,
            model: None,
        }
    }
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }
}

#[async_trait]
impl CompletionVerifier for LlmJudgeVerifier {
    async fn verify(&self, request: &CompletionRequest, answer: &str) -> Verification {
        let judge_request = CompletionRequest::new(format!(
            "Evaluate whether the candidate answer correctly and completely answers the user request. Reply with exactly APPROVE or REJECT, then a short reason.\n\nUSER REQUEST:\n{}\n\nCANDIDATE ANSWER:\n{}", request.prompt, answer
        )).model(self.model.clone().unwrap_or_else(|| "default".into()));
        match self.provider.complete(&judge_request).await {
            Ok(result) => {
                let input = result.usage.prompt_tokens as f64 / 1_000.0
                    * self
                        .provider
                        .input_cost_per_1k(&result.model)
                        .unwrap_or(0.0);
                let output = result.usage.completion_tokens as f64 / 1_000.0
                    * self
                        .provider
                        .output_cost_per_1k(&result.model)
                        .unwrap_or(0.0);
                let total = input + output;
                Verification {
                    approved: result
                        .text
                        .trim_start()
                        .to_ascii_uppercase()
                        .starts_with("APPROVE"),
                    provider: self.provider.name().into(),
                    reason: result.text,
                    cost: CostSummary {
                        input_usd: input,
                        output_usd: output,
                        verification_usd: total,
                        total_usd: total,
                        avoided_usd: 0.0,
                    },
                }
            }
            Err(error) => Verification {
                approved: false,
                provider: self.provider.name().into(),
                reason: format!("verifier unavailable: {error}"),
                cost: CostSummary::default(),
            },
        }
    }
}
