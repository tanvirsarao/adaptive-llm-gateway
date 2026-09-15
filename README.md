<div align="center">
  <h1>Adaptive LLM Gateway</h1>
  <p><strong>A tiny Rust control plane for cheaper, faster, provider-agnostic LLM inference.</strong></p>
  <p><a href="#try-it-in-60-seconds">Try it</a> · <a href="#how-it-works">How it works</a> · <a href="#use-it-as-an-sdk">Rust SDK</a> · <a href="#roadmap">Roadmap</a></p>
  <p>
    <img src="https://img.shields.io/badge/Rust-1.82%2B-DEA584?logo=rust&logoColor=white" alt="Rust 1.82+" />
    <img src="https://img.shields.io/badge/API-Axum-7A3E9D" alt="Axum API" />
    <img src="https://img.shields.io/badge/Cache-exact%20%2B%20semantic-00BFA6" alt="Exact and semantic caching" />
    <img src="https://img.shields.io/badge/License-MIT-4C1" alt="MIT License" />
  </p>
</div>

<br />

> **The idea:** your application makes one LLM call. The gateway picks the lowest-cost capable provider, falls back when needed, and never pays twice for work it has already seen.

```
your app
   │  POST /v1/chat/completions
   ▼
┌──────────────────────────────────────────────────────────────┐
│                    ADAPTIVE LLM GATEWAY                       │
│                                                              │
│   exact cache ──► semantic cache ──► cost-aware router       │
│      Redis             pgvector          ↓                  │
└─────────────────────────────────────────┼────────────────────┘
                                          ▼
                            OpenAI · Anthropic · local models
```

## Why this exists

Most apps begin with a single model and a single API key. That is wonderfully simple—until cost, outages, and repeated requests start to matter. Adaptive LLM Gateway is an intentionally small layer between your app and model providers. Its job is to make the sensible default automatic:

- **Route economically.** Select the compatible provider with the lowest configured token cost.
- **Cache intelligently.** Identical requests are exact hits; near-identical requests can be served by a semantic cache when you provide embeddings.
- **Fail gracefully.** Retryable provider failures fall through to the next best route.
- **Stay portable.** Embed it as a Rust SDK today or expose the same gateway behind an HTTP API.

## Try it in 60 seconds

The demo ships with a local provider. It requires **no API key, Redis, Postgres, or model download**.

```bash
git clone https://github.com/tanvirsarao/adaptive-llm-gateway.git
cd adaptive-llm-gateway
cargo run --features server
```

In a second terminal, make the same request twice:

```bash
curl -s http://127.0.0.1:3000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"prompt":"Explain cache invalidation in six words."}' | jq
```

First response:

```json
{
  "text": "Demo answer: Explain cache invalidation in six words.",
  "provider": "local-demo",
  "cached": false,
  "cache_kind": null
}
```

Run it once more. The gateway returns the response without calling a provider:

```json
{
  "provider": "local-demo",
  "cached": true,
  "cache_kind": "exact"
}
```

Try a semantic-cache lookup by including an embedding. Two requests with close vectors reuse the first answer once their cosine similarity reaches the configured threshold (default: `0.96`).

## Use it as an SDK

The HTTP server is only a wrapper. The primary interface is an embeddable Rust gateway, which makes it easy to keep routing close to your application code.

```rust
use std::sync::Arc;
use adaptive_llm_gateway::{CompletionRequest, Gateway, GatewayConfig, InMemoryCache};

# async fn example() -> Result<(), adaptive_llm_gateway::GatewayError> {
let gateway = Gateway::builder(GatewayConfig::default())
    .cache(Arc::new(InMemoryCache::default()))
    .provider(Arc::new(my_openai_provider))
    .provider(Arc::new(my_local_model_provider))
    .build()?;

let answer = gateway
    .complete(
        CompletionRequest::new("Summarize this support ticket".into())
            .model("default")
            .embedding(ticket_embedding),
    )
    .await?;

println!("{} via {} (cached: {})", answer.text, answer.provider, answer.cached);
# Ok(()) }
```

Implementing a provider is deliberately small: give it a name, a list of models, a cost estimate, and one `complete` method. That keeps OpenAI-compatible APIs, Anthropic, vLLM, Ollama, and internal model endpoints equally possible.

## HTTP API

`POST /v1/chat/completions`

```json
{
  "prompt": "Write a warm release note.",
  "model": "default",
  "max_tokens": 200,
  "temperature": 0.7,
  "embedding": [0.12, -0.38, 0.91]
}
```

Every response includes the routing decision and cache status, making it straightforward to emit metrics or build an internal cost dashboard.

| Field | Meaning |
| --- | --- |
| `provider` | The provider which generated the response (or originally generated a cached response). |
| `model` | The selected model. |
| `cached` | Whether no provider call was made. |
| `cache_kind` | `exact`, `semantic`, or `null`. |
| `usage` | Prompt and completion token counts reported by the provider. |

## How it works

For each completion, the gateway follows a short, predictable path:

1. Create a stable cache key from the prompt and generation settings.
2. Return an exact cache hit if one exists.
3. If an embedding was supplied, return the closest semantic match above the threshold.
4. Rank compatible providers by configured input-token cost.
5. Call the best candidate and retry the next candidate only for retryable failures.
6. Store the completed response for future exact and semantic lookups.

The included `InMemoryCache` makes the demo self-contained. The `Cache` trait is the production seam: back exact keys with Redis and `find_similar` with pgvector's cosine-distance query. The gateway does not own embedding generation, so you can use the embedding model and privacy boundary that fit your system.

## Roadmap

This is a focused MVP, optimized for a credible local demo and a clean integration surface—not a claim that every provider integration is already production-ready.

**Working now**

- Cost-ordered provider selection and retryable fallback
- Exact cache and in-memory semantic cosine search
- Rust SDK surface and Axum demo API
- Structured tracing hooks for provider, cache, and latency events

**Next up**

- Redis exact-cache adapter with TTL and namespacing
- pgvector cache adapter and embedding-provider helpers
- OpenAI, Anthropic, Ollama, and OpenAI-compatible provider crates
- OpenTelemetry spans and Prometheus metrics
- Streaming responses, budgets, rate limits, and tenant-aware routing

## Development

```bash
cargo fmt --check
cargo test
cargo clippy --all-targets --all-features -- -D warnings
```

This is a personal project. The roadmap is intentionally scoped around the parts that make it useful in real applications: local inference, reliable caching, and transparent routing decisions.
