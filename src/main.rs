use adaptive_llm_gateway::{
    CompletionRequest, Gateway, GatewayConfig, InMemoryCache, LlmProvider, ProviderCompletion,
    ProviderError, Usage,
};
use async_trait::async_trait;
use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::Serialize;
use std::{sync::Arc, time::Instant};

struct DemoProvider {
    name: String,
    models: Vec<String>,
    price: f64,
}
#[async_trait]
impl LlmProvider for DemoProvider {
    fn name(&self) -> &str {
        &self.name
    }
    fn models(&self) -> &[String] {
        &self.models
    }
    fn input_cost_per_1k(&self, _: &str) -> Option<f64> {
        Some(self.price)
    }
    async fn complete(
        &self,
        request: &CompletionRequest,
    ) -> Result<ProviderCompletion, ProviderError> {
        Ok(ProviderCompletion {
            text: format!("Demo answer: {}", request.prompt),
            model: request.model.clone().unwrap_or_else(|| "default".into()),
            usage: Usage {
                prompt_tokens: request.prompt.split_whitespace().count() as u32,
                completion_tokens: 8,
            },
        })
    }
}

#[derive(Serialize)]
struct Health {
    status: &'static str,
}
async fn health() -> Json<Health> {
    Json(Health { status: "ok" })
}
async fn complete(
    State(gateway): State<Arc<Gateway>>,
    Json(request): Json<CompletionRequest>,
) -> Result<Json<adaptive_llm_gateway::Completion>, (StatusCode, String)> {
    let started = Instant::now();
    let response = gateway
        .complete(request)
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))?;
    tracing::info!(
        latency_ms = started.elapsed().as_millis() as u64,
        cached = response.cached,
        "request completed"
    );
    Ok(Json(response))
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_target(false)
        .compact()
        .init();
    let provider = DemoProvider {
        name: "local-demo".into(),
        models: vec!["default".into(), "demo".into()],
        price: 0.0001,
    };
    let gateway = Arc::new(
        Gateway::builder(GatewayConfig::default())
            .cache(Arc::new(InMemoryCache::default()))
            .provider(Arc::new(provider))
            .build()
            .expect("valid gateway"),
    );
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/chat/completions", post(complete))
        .with_state(gateway);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000")
        .await
        .expect("bind port 3000");
    tracing::info!("adaptive-llm-gateway demo listening on http://127.0.0.1:3000");
    axum::serve(listener, app).await.expect("server error");
}
