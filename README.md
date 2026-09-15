<div align="center">
  <h1>Adaptive LLM Gateway</h1>
  <p><strong>Route every prompt through the best model. See every decision.</strong></p>
  <p>
    <a href="#try-the-demo">Try the demo</a> ·
    <a href="#run-a-real-hugging-face-model-locally">Run local inference</a> ·
    <a href="#use-it-as-an-sdk">Use the SDK</a>
  </p>
  <p>
    <img src="https://img.shields.io/badge/Rust-1.82%2B-DEA584?logo=rust&logoColor=white" alt="Rust 1.82+" />
    <img src="https://img.shields.io/badge/CLI-explainable%20routing-7A3E9D" alt="Explainable CLI" />
    <img src="https://img.shields.io/badge/Cache-exact%20%2B%20semantic-00BFA6" alt="Exact and semantic caching" />
    <img src="https://img.shields.io/badge/Local%20models-Hugging%20Face-FFD21E?logo=huggingface&logoColor=black" alt="Hugging Face local models" />
  </p>
</div>

<br />

> **One prompt in. A cheaper, observable inference decision out.** Adaptive LLM Gateway ranks providers by cost, falls back safely, caches repeated work, and tells you exactly what happened.

```
your application or terminal
            │
            ▼
┌───────────────────────────────────────────────────────────┐
│                  ADAPTIVE LLM GATEWAY                      │
│                                                           │
│  exact cache ──► semantic cache ──► cost-aware router     │
│       ↓                 ↓                   ↓             │
│     $0 reply          close match      local → frontier   │
└───────────────────────────────────────────────────────────┘
```

## Try the demo

No model download. No API key. No Redis or Postgres. The demo intentionally makes the free local route unavailable, so you can see a paid fallback, its token cost, and then a cache replay that costs nothing.

```bash
git clone https://github.com/tanvirsarao/adaptive-llm-gateway.git
cd adaptive-llm-gateway
cargo run --features cli --bin gateway -- demo \
  "Explain cache invalidation in six words." \
  --simulate-local-failure
```

The CLI is the product demo—not a hidden debug log:

```text
  ADAPTIVE LLM GATEWAY  ·  demo

  ── FIRST REQUEST ─────────────────────────────────────
  Cache     MISS
  Routes
    1. local-qwen             default   in $0.0000/1K · out $0.0000/1K
    2. frontier-fallback      default   in $0.5000/1K · out $3.0000/1K
  Attempt   local-qwen → retryable failure: simulated local endpoint unavailable
  Attempt   frontier-fallback → selected
  Response  Demo answer: Explain cache invalidation in six words.
  Tokens    6 input · 18 output
  Cost      $0.057000 this request

  Replaying the same request to show the cache…

  ── REPLAY ────────────────────────────────────────────
  Cache     EXACT
  Tokens    6 input · 18 output
  Cost      $0.000000 this request
  Saved     $0.057000 via exact cache
```

Run without `--simulate-local-failure` to see the gateway keep the request on the no-cost local route. The CLI shows candidate routes, the selected provider, failures/fallbacks, token counts, spend, and cache savings for every request.

## Run a real Hugging Face model locally

The gateway speaks the OpenAI-compatible chat API, so it works with vLLM, LM Studio, and llama.cpp. For the shortest cross-platform path, use llama.cpp: it can download a quantized GGUF model directly from Hugging Face and expose a local API. The [llama.cpp quick start](https://github.com/ggml-org/llama.cpp#quick-start) documents the `llama serve -hf …` workflow and its OpenAI-compatible server.

1. Install a current llama.cpp build (prebuilt release or source build) so the `llama` command is available.

2. In one terminal, download and serve a compact Hugging Face model. The first launch downloads it; later launches use the local cache.

   ```bash
   llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF --alias local-qwen
   ```

3. In this repository, ask the gateway to use it:

   ```bash
   cargo run --features cli --bin gateway -- prompt \
     "Give me three names for a coffee shop for night owls." \
     --endpoint http://127.0.0.1:8080/v1 \
     --model local-qwen \
     --replay
   ```

`--replay` sends the same prompt again in the same gateway process, making the exact-cache hit visible. A local model defaults to `$0.00` pricing. If you are routing to a paid OpenAI-compatible endpoint, add `--input-cost-per-1k` and `--output-cost-per-1k` to have the CLI calculate actual request spend.

For a GPU server, point the same command at vLLM instead. vLLM serves Hugging Face model IDs behind `/v1/chat/completions`; see its [OpenAI-compatible server documentation](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/).

## Why this exists

Apps usually start with one model and one API key. That is simple until cost, outages, and repeated requests matter. This gateway is a small layer between your application and the inference endpoints that makes the sensible default automatic:

- **Route economically.** Compatible providers are ranked by configured token cost.
- **Cache intelligently.** Identical prompts are exact hits; near-identical prompts can reuse a semantic match when an embedding is supplied.
- **Fail gracefully.** A retryable provider failure moves to the next best route.
- **Explain itself.** Every completion includes the candidates, attempts, token use, spend, and avoided cache cost.

## Use it as an SDK

The CLI and HTTP API are thin clients over the same Rust gateway. Embed it when you want routing close to your application.

```rust
use std::sync::Arc;
use adaptive_llm_gateway::{
    CompletionRequest, Gateway, GatewayConfig, InMemoryCache, OpenAiCompatibleProvider,
};

# async fn example() -> Result<(), adaptive_llm_gateway::GatewayError> {
let local_qwen = OpenAiCompatibleProvider::new(
    "local-qwen",
    "http://127.0.0.1:8080/v1",
    vec!["local-qwen".into()],
);

let gateway = Gateway::builder(GatewayConfig::default())
    .cache(Arc::new(InMemoryCache::default()))
    .provider(Arc::new(local_qwen))
    .build()?;

let answer = gateway
    .complete(CompletionRequest::new("Summarize this support ticket".into()).model("local-qwen"))
    .await?;

println!("{} via {} — ${:.6}", answer.text, answer.provider, answer.cost.total_usd);
# Ok(()) }
```

Enable the provider adapter in your application's dependency declaration:

```toml
adaptive-llm-gateway = { version = "0.1", features = ["openai-compatible"] }
```

## HTTP API

The optional Axum server exposes the same completion object at `POST /v1/chat/completions`.

```bash
cargo run --features server
curl -s http://127.0.0.1:3000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"prompt":"Write a warm release note."}'
```

Alongside the response text, the JSON includes `usage`, `cost`, and `route` so a caller can persist or chart the same decision data shown by the CLI.

## How it works

For every prompt, the gateway:

1. Checks the exact cache.
2. Checks the semantic cache when an embedding is supplied.
3. Ranks compatible providers by estimated input-token cost.
4. Calls the best route, falling through only after retryable failures.
5. Records the response for future exact and semantic lookups.

`InMemoryCache` keeps the demo self-contained. The `Cache` trait is the production seam: Redis is a natural exact-cache implementation, while pgvector can implement similarity search with cosine distance. The gateway intentionally leaves embedding generation to the application so its privacy boundary and embedding model stay under your control.

## Project status

**Working now**

- Explainable CLI walkthrough with fallback, token counts, costs, and cache savings
- OpenAI-compatible local/hosted provider adapter
- Exact cache and in-memory semantic cosine search
- Cost-ordered routing with retryable fallback
- Optional Axum API and structured tracing hooks

**Next up**

- Redis cache adapter with TTL and namespacing
- pgvector semantic-cache adapter and embedding helpers
- Streaming responses, budgets, rate limits, and tenant-aware routing
- OpenTelemetry spans and Prometheus metrics

## Development

```bash
cargo fmt --check
cargo test --all-features
cargo clippy --all-targets --all-features -- -D warnings
```
