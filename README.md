<div align="center">
  <h1>Adaptive LLM Gateway</h1>
  <p><strong>Model-agnostic inference memory and intelligent LLM delegation.</strong></p>
  <p>
    <a href="#architecture">Architecture</a> ·
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

> **One prompt in. The cheapest trustworthy answer out.** Adaptive LLM Gateway remembers inference across providers, delegates work to cheap specialists first, and escalates to frontier models only when a verifier says it must.

```
your application or terminal
            │
            ▼
┌───────────────────────────────────────────────────────────┐
│                  ADAPTIVE LLM GATEWAY                      │
│                                                           │
│   L1 exact ──► L2 semantic ──► L3 prefix/KV reuse         │
│    Redis          pgvector        provider runtime         │
│      │               │                    │               │
│      └──── no inference ────┐    fresh answer, less work  │
│                              ▼                             │
│          cheap specialist → verifier → frontier escalation │
└───────────────────────────────────────────────────────────┘
```

## Architecture

The gateway is built around two connected systems: **a cost-aware delegation cascade** and **a provider-independent inference memory**.

### 1. Delegate cheap work; reserve frontier reasoning

Execution providers are ranked by cost. A cheap/local model drafts the answer first. An optional verifier accepts the draft or rejects it; only a rejection moves the request to the next, more capable route. The verifier can itself be a frontier model running a strict `APPROVE`/`REJECT` check.

That makes the expensive model a selective judge and escalation path, not the default answer engine. The final response preserves every attempted provider, verifier decision, token count, latency hook, and dollar cost—including the cost of verification.

### 2. A three-layer inference memory

| Layer | Store | What is reused | Outcome |
| --- | --- | --- | --- |
| L1 — exact response | Redis | Canonical request → complete response | No model inference |
| L2 — semantic response | PostgreSQL + pgvector | Safe, compatible near-match → response | No model inference |
| L3 — prefix/KV | Local/provider runtime | Attention state for shared instructions/context | Fresh answer with less prefill work |

L1 and L2 survive model changes because they store an answer contract, not provider-specific internal state. L3 cannot transfer between model architectures; it is intentionally reported as provider-native reuse rather than pretending every LLM shares the same KV cache.

### 3. Cache correctness before cache hit rate

An exact key is a versioned SHA-256 fingerprint of the prompt, system instructions, selected model, generation settings, tenant scope, and cache policy. A semantic hit is additionally gated by tenant scope, output/task compatibility key, expiry, and a caller-selected `SafeReadOnly` policy. Embeddings choose a semantic candidate—they never alter exact-cache identity.

This means “similar” does not silently become “reusable.” User-specific, mutable, tool-using, structured, or high-stakes work can keep semantic reuse disabled while still benefiting from exact and provider-native prefix caching.

For more implementation detail, see [the architecture notes](docs/architecture.md).

## Persistent cache setup

Enable both durable cache tiers in your application:

```toml
adaptive-llm-gateway = { version = "0.1", features = [
  "openai-compatible", "redis-cache", "pgvector-cache"
] }
```

`RedisExactCache` owns L1 keys with a namespace and TTL. `PgVectorSemanticCache` owns L2 embeddings and must be initialized through your normal database migration process using `PgVectorSemanticCache::SCHEMA_SQL`. Combine them with `TieredCache`; align its tenant scope and compatibility key with the request’s `cache_scope` and `cache_policy`.

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

- **Delegate economically.** Cheap specialists draft; a verifier can escalate only the hard work.
- **Remember inference.** Exact, semantic, and native-prefix layers eliminate or reduce repeated compute.
- **Stay consistent across models.** Response reuse is protected by scopes, request contracts, compatibility policies, and expiry.
- **Explain itself.** Every completion includes candidates, attempts, verifier decisions, token use, spend, and avoided cost.

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

1. Canonicalizes the complete request contract and checks L1 exact memory.
2. Checks L2 semantic memory only if the caller explicitly permits safe approximate reuse.
3. Ranks eligible execution providers by estimated cost.
4. Lets the cheapest provider draft, then optionally verifies its answer.
5. Escalates rejected drafts or retryable failures to the next provider.
6. Records the final answer and its route/cost provenance for future reuse.

`InMemoryCache` keeps the demo self-contained. `RedisExactCache` and `PgVectorSemanticCache` provide the durable production tiers. The gateway intentionally leaves embedding generation to the application so its privacy boundary and embedding model stay under your control.

## Project status

**Working now**

- Delegation cascade with optional LLM judge and cost-accounted escalation
- Explainable CLI walkthrough with fallback, token counts, costs, and cache savings
- OpenAI-compatible local/hosted provider adapter
- Redis L1 exact cache and pgvector L2 semantic cache
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
