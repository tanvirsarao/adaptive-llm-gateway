use crate::{
    Cache, CacheEntry, Completion, CompletionRequest, CompletionVerifier, CostSummary,
    GatewayError, LlmProvider, ProviderAttempt, RouteCandidate, RouteTrace, SemanticReuse,
};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use tracing::{info, warn};
use uuid::Uuid;

/// Behavioural settings for a [`Gateway`].
#[derive(Clone, Debug)]
pub struct GatewayConfig {
    /// Reuse embeddings whose cosine similarity meets this threshold.
    pub semantic_cache_threshold: f32,
    /// Use a semantic hit only when the request supplies an embedding.
    pub semantic_cache_enabled: bool,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            semantic_cache_threshold: 0.96,
            semantic_cache_enabled: true,
        }
    }
}

pub struct GatewayBuilder {
    config: GatewayConfig,
    providers: Vec<Arc<dyn LlmProvider>>,
    cache: Option<Arc<dyn Cache>>,
    verifier: Option<Arc<dyn CompletionVerifier>>,
}

impl GatewayBuilder {
    pub fn provider(mut self, provider: Arc<dyn LlmProvider>) -> Self {
        self.providers.push(provider);
        self
    }
    pub fn cache(mut self, cache: Arc<dyn Cache>) -> Self {
        self.cache = Some(cache);
        self
    }
    /// Rejecting a draft makes the gateway continue to the next cost-ranked provider.
    pub fn verifier(mut self, verifier: Arc<dyn CompletionVerifier>) -> Self {
        self.verifier = Some(verifier);
        self
    }
    pub fn build(self) -> Result<Gateway, GatewayError> {
        if self.providers.is_empty() {
            return Err(GatewayError::Configuration(
                "configure at least one provider".into(),
            ));
        }
        Ok(Gateway {
            config: self.config,
            providers: self.providers,
            cache: self.cache,
            verifier: self.verifier,
        })
    }
}

/// Routes each request to the lowest estimated-cost compatible provider, and
/// falls back to the next one if it returns a retryable error.
pub struct Gateway {
    config: GatewayConfig,
    providers: Vec<Arc<dyn LlmProvider>>,
    cache: Option<Arc<dyn Cache>>,
    verifier: Option<Arc<dyn CompletionVerifier>>,
}

impl Gateway {
    pub fn builder(config: GatewayConfig) -> GatewayBuilder {
        GatewayBuilder {
            config,
            providers: Vec::new(),
            cache: None,
            verifier: None,
        }
    }

    pub async fn complete(&self, request: CompletionRequest) -> Result<Completion, GatewayError> {
        let key = cache_key(&request);
        if let Some(cache) = &self.cache {
            if let Some(mut entry) = cache.get_exact(&key).await.map_err(GatewayError::Cache)? {
                let avoided = entry.completion.cost.total_usd;
                entry.completion.cached = true;
                entry.completion.cache_kind = Some("exact".into());
                entry.completion.cost = CostSummary {
                    avoided_usd: avoided,
                    ..Default::default()
                };
                entry.completion.route = RouteTrace {
                    cache: "exact".into(),
                    ..Default::default()
                };
                info!(cache = "exact", completion_id = %entry.completion.id, "gateway cache hit");
                return Ok(entry.completion);
            }
            if self.config.semantic_cache_enabled
                && request.cache_policy.semantic_reuse == SemanticReuse::SafeReadOnly
            {
                if let Some(embedding) = &request.embedding {
                    if let Some(mut hit) = cache
                        .find_similar(embedding, self.config.semantic_cache_threshold)
                        .await
                        .map_err(GatewayError::Cache)?
                    {
                        let avoided = hit.entry.completion.cost.total_usd;
                        hit.entry.completion.cached = true;
                        hit.entry.completion.cache_kind = Some("semantic".into());
                        hit.entry.completion.cost = CostSummary {
                            avoided_usd: avoided,
                            ..Default::default()
                        };
                        hit.entry.completion.route = RouteTrace {
                            cache: "semantic".into(),
                            semantic_similarity: Some(hit.similarity),
                            ..Default::default()
                        };
                        info!(cache = "semantic", similarity = hit.similarity, completion_id = %hit.entry.completion.id, "gateway cache hit");
                        return Ok(hit.entry.completion);
                    }
                }
            }
        }

        let model = request.model.clone().unwrap_or_else(|| "default".into());
        let mut candidates: Vec<_> = self
            .providers
            .iter()
            .filter(|p| p.models().iter().any(|m| m == &model || m == "default"))
            .collect();
        candidates.sort_by(|a, b| {
            a.input_cost_per_1k(&model)
                .unwrap_or(f64::INFINITY)
                .total_cmp(&b.input_cost_per_1k(&model).unwrap_or(f64::INFINITY))
        });
        if candidates.is_empty() {
            return Err(GatewayError::NoProvider(model));
        }
        let route_candidates = candidates
            .iter()
            .map(|provider| RouteCandidate {
                provider: provider.name().into(),
                model: model.clone(),
                input_cost_per_1k_usd: provider.input_cost_per_1k(&model).unwrap_or(f64::INFINITY),
                output_cost_per_1k_usd: provider
                    .output_cost_per_1k(&model)
                    .unwrap_or(f64::INFINITY),
            })
            .collect();

        let mut errors = Vec::new();
        let mut attempts = Vec::new();
        let mut verification_cost = 0.0;
        for provider in candidates {
            match provider.complete(&request).await {
                Ok(result) => {
                    let input_usd = result.usage.prompt_tokens as f64 / 1_000.0
                        * provider.input_cost_per_1k(&model).unwrap_or(0.0);
                    let output_usd = result.usage.completion_tokens as f64 / 1_000.0
                        * provider.output_cost_per_1k(&model).unwrap_or(0.0);
                    if let Some(verifier) = &self.verifier {
                        let verification = verifier.verify(&request, &result.text).await;
                        verification_cost += verification.cost.total_usd;
                        attempts.push(ProviderAttempt {
                            provider: verification.provider,
                            outcome: format!("verification: {}", verification.reason),
                        });
                        if !verification.approved {
                            attempts.push(ProviderAttempt {
                                provider: provider.name().into(),
                                outcome: "draft rejected; escalating".into(),
                            });
                            continue;
                        }
                    }
                    attempts.push(ProviderAttempt {
                        provider: provider.name().into(),
                        outcome: "selected".into(),
                    });
                    let completion = Completion {
                        id: Uuid::new_v4(),
                        text: result.text,
                        model: result.model,
                        provider: provider.name().into(),
                        usage: result.usage,
                        cached: false,
                        cache_kind: None,
                        cost: CostSummary {
                            input_usd,
                            output_usd,
                            verification_usd: verification_cost,
                            total_usd: input_usd + output_usd + verification_cost,
                            avoided_usd: 0.0,
                        },
                        route: RouteTrace {
                            cache: "miss".into(),
                            semantic_similarity: None,
                            candidates: route_candidates,
                            attempts,
                        },
                    };
                    if let Some(cache) = &self.cache {
                        cache
                            .put_exact(
                                key,
                                CacheEntry {
                                    completion: completion.clone(),
                                    embedding: request.embedding.clone(),
                                    policy: request.cache_policy.clone(),
                                },
                            )
                            .await
                            .map_err(GatewayError::Cache)?;
                    }
                    info!(provider = provider.name(), completion_id = %completion.id, "gateway routed request");
                    return Ok(completion);
                }
                Err(error) if error.retryable => {
                    warn!(provider = provider.name(), error = %error, "provider failed; trying fallback");
                    errors.push(format!("{}: {}", provider.name(), error));
                    attempts.push(ProviderAttempt {
                        provider: provider.name().into(),
                        outcome: format!("retryable failure: {error}"),
                    });
                }
                Err(error) => {
                    return Err(GatewayError::ProvidersFailed(format!(
                        "{}: {}",
                        provider.name(),
                        error
                    )))
                }
            }
        }
        Err(GatewayError::ProvidersFailed(errors.join("; ")))
    }
}

fn cache_key(request: &CompletionRequest) -> String {
    #[derive(serde::Serialize)]
    struct CanonicalRequest<'a> {
        cache_schema: &'static str,
        prompt: String,
        system: Option<String>,
        model: &'a Option<String>,
        max_tokens: &'a Option<u32>,
        temperature_bits: Option<u32>,
        cache_scope: &'a str,
        policy: &'a crate::CachePolicy,
    }
    let canonical = CanonicalRequest {
        cache_schema: "request:v2",
        prompt: request.prompt.replace("\r\n", "\n"),
        system: request
            .system
            .as_ref()
            .map(|value| value.replace("\r\n", "\n")),
        model: &request.model,
        max_tokens: &request.max_tokens,
        temperature_bits: request.temperature.map(f32::to_bits),
        cache_scope: &request.cache_scope,
        policy: &request.cache_policy,
    };
    let bytes = serde_json::to_vec(&canonical).expect("canonical cache request serializes");
    let digest = Sha256::digest(bytes);
    format!("v2:{digest:x}")
}
