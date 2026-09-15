use adaptive_llm_gateway::{
    CompletionRequest, CompletionVerifier, CostSummary, Gateway, GatewayConfig, InMemoryCache,
    LlmProvider, ProviderCompletion, ProviderError, Usage, Verification,
};
use async_trait::async_trait;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

struct Fake {
    calls: AtomicUsize,
    name: String,
    cost: f64,
}

struct RejectCheap;
#[async_trait]
impl CompletionVerifier for RejectCheap {
    async fn verify(&self, _: &CompletionRequest, answer: &str) -> Verification {
        Verification {
            approved: answer == "frontier",
            provider: "judge".into(),
            reason: if answer == "frontier" {
                "APPROVE"
            } else {
                "REJECT"
            }
            .into(),
            cost: CostSummary {
                verification_usd: 0.01,
                total_usd: 0.01,
                ..Default::default()
            },
        }
    }
}
#[async_trait]
impl LlmProvider for Fake {
    fn name(&self) -> &str {
        &self.name
    }
    fn models(&self) -> &[String] {
        static MODELS: std::sync::LazyLock<Vec<String>> =
            std::sync::LazyLock::new(|| vec!["default".into()]);
        &MODELS
    }
    fn input_cost_per_1k(&self, _: &str) -> Option<f64> {
        Some(self.cost)
    }
    async fn complete(&self, _: &CompletionRequest) -> Result<ProviderCompletion, ProviderError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(ProviderCompletion {
            text: self.name.clone(),
            model: "default".into(),
            usage: Usage {
                prompt_tokens: 1,
                completion_tokens: 1,
            },
        })
    }
}

#[tokio::test]
async fn caches_a_repeat_request() {
    let provider = Arc::new(Fake {
        calls: AtomicUsize::new(0),
        name: "cheap".into(),
        cost: 0.1,
    });
    let gateway = Gateway::builder(GatewayConfig::default())
        .cache(Arc::new(InMemoryCache::default()))
        .provider(provider.clone())
        .build()
        .unwrap();
    gateway
        .complete(CompletionRequest::new("hello".into()))
        .await
        .unwrap();
    let second = gateway
        .complete(CompletionRequest::new("hello".into()))
        .await
        .unwrap();
    assert!(second.cached);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn selects_the_lower_cost_provider() {
    let expensive = Arc::new(Fake {
        calls: AtomicUsize::new(0),
        name: "expensive".into(),
        cost: 2.0,
    });
    let cheap = Arc::new(Fake {
        calls: AtomicUsize::new(0),
        name: "cheap".into(),
        cost: 0.1,
    });
    let gateway = Gateway::builder(GatewayConfig::default())
        .provider(expensive.clone())
        .provider(cheap.clone())
        .build()
        .unwrap();
    assert_eq!(
        gateway
            .complete(CompletionRequest::new("hello".into()))
            .await
            .unwrap()
            .provider,
        "cheap"
    );
    assert_eq!(expensive.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn rejected_cheap_draft_escalates_and_keeps_verification_cost() {
    let cheap = Arc::new(Fake {
        calls: AtomicUsize::new(0),
        name: "cheap".into(),
        cost: 0.1,
    });
    let frontier = Arc::new(Fake {
        calls: AtomicUsize::new(0),
        name: "frontier".into(),
        cost: 2.0,
    });
    let gateway = Gateway::builder(GatewayConfig::default())
        .provider(cheap)
        .provider(frontier)
        .verifier(Arc::new(RejectCheap))
        .build()
        .unwrap();
    let response = gateway
        .complete(CompletionRequest::new("hello".into()))
        .await
        .unwrap();
    assert_eq!(response.provider, "frontier");
    assert_eq!(response.cost.verification_usd, 0.02);
    assert!(response
        .route
        .attempts
        .iter()
        .any(|attempt| attempt.outcome == "draft rejected; escalating"));
}
