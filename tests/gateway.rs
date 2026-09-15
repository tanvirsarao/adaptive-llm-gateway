use adaptive_llm_gateway::{
    CompletionRequest, Gateway, GatewayConfig, InMemoryCache, LlmProvider, ProviderCompletion,
    ProviderError, Usage,
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
