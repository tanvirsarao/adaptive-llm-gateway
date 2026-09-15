use adaptive_llm_gateway::{
    CachePolicy, Completion, CompletionRequest, Gateway, GatewayConfig, InMemoryCache, LlmProvider,
    OpenAiCompatibleProvider, ProviderCompletion, ProviderError, SemanticReuse, Usage,
};
use async_trait::async_trait;
use clap::{Args, Parser, Subcommand};
use std::{io::IsTerminal, sync::Arc};

#[derive(Parser)]
#[command(
    name = "gateway",
    version,
    about = "See and control every LLM routing decision."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run a zero-key routing, cost, and cache walkthrough.
    Demo {
        prompt: String,
        #[arg(long)]
        simulate_local_failure: bool,
    },
    /// Send one prompt to a real OpenAI-compatible local server (vLLM, LM Studio, llama.cpp).
    Prompt(PromptArgs),
}

#[derive(Args)]
struct PromptArgs {
    prompt: String,
    /// API root, including /v1.
    #[arg(long, default_value = "http://127.0.0.1:8000/v1")]
    endpoint: String,
    /// Must match the name configured in your local inference server.
    #[arg(long, default_value = "Qwen/Qwen3-4B-Instruct-2507")]
    model: String,
    #[arg(long, default_value_t = 0.0)]
    input_cost_per_1k: f64,
    #[arg(long, default_value_t = 0.0)]
    output_cost_per_1k: f64,
    /// Repeat once in the same process to demonstrate an exact cache hit.
    #[arg(long)]
    replay: bool,
}

struct ShowcaseProvider {
    name: String,
    model: String,
    input_cost: f64,
    output_cost: f64,
    fail: bool,
}
#[async_trait]
impl LlmProvider for ShowcaseProvider {
    fn name(&self) -> &str {
        &self.name
    }
    fn models(&self) -> &[String] {
        std::slice::from_ref(&self.model)
    }
    fn input_cost_per_1k(&self, _: &str) -> Option<f64> {
        Some(self.input_cost)
    }
    fn output_cost_per_1k(&self, _: &str) -> Option<f64> {
        Some(self.output_cost)
    }
    async fn complete(
        &self,
        request: &CompletionRequest,
    ) -> Result<ProviderCompletion, ProviderError> {
        if self.fail {
            return Err(ProviderError {
                message: "simulated local endpoint unavailable".into(),
                retryable: true,
            });
        }
        Ok(ProviderCompletion {
            text: format!("Demo answer: {}", request.prompt),
            model: self.model.clone(),
            usage: Usage {
                prompt_tokens: request.prompt.split_whitespace().count() as u32,
                completion_tokens: 18,
            },
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    match Cli::parse().command {
        Command::Demo {
            prompt,
            simulate_local_failure,
        } => demo(prompt, simulate_local_failure).await?,
        Command::Prompt(args) => prompt(args).await?,
    }
    Ok(())
}

async fn demo(prompt: String, local_fails: bool) -> Result<(), Box<dyn std::error::Error>> {
    let local = ShowcaseProvider {
        name: "local-qwen".into(),
        model: "default".into(),
        input_cost: 0.0,
        output_cost: 0.0,
        fail: local_fails,
    };
    let frontier = ShowcaseProvider {
        name: "frontier-fallback".into(),
        model: "default".into(),
        input_cost: 0.50,
        output_cost: 3.00,
        fail: false,
    };
    let gateway = Gateway::builder(GatewayConfig::default())
        .cache(Arc::new(InMemoryCache::default()))
        .provider(Arc::new(local))
        .provider(Arc::new(frontier))
        .build()?;
    let theme = Theme::detect();
    println!(
        "\n  {}  {}\n",
        theme.brand("ADAPTIVE LLM GATEWAY"),
        theme.muted("· demo")
    );
    let semantic_policy = CachePolicy {
        compatibility_key: "demo:read-only:v1".into(),
        semantic_reuse: SemanticReuse::SafeReadOnly,
    };
    let request = CompletionRequest::new(prompt.clone())
        .embedding(vec![0.12, -0.38, 0.91])
        .cache_policy(semantic_policy.clone());
    let first = gateway.complete(request.clone()).await?;
    print_completion("FIRST REQUEST", &first, theme);
    println!(
        "\n  {}",
        theme.muted("Replaying the same request to show the exact cache…")
    );
    let replay = gateway.complete(request).await?;
    print_completion("EXACT REPLAY", &replay, theme);
    println!(
        "\n  {}",
        theme.muted("Trying a near-match prompt to show the semantic cache…")
    );
    let semantic = gateway
        .complete(
            CompletionRequest::new(format!("{prompt} Please keep it concise."))
                .embedding(vec![0.12, -0.38, 0.91])
                .cache_policy(semantic_policy),
        )
        .await?;
    print_completion("SEMANTIC REPLAY", &semantic, theme);
    Ok(())
}

async fn prompt(args: PromptArgs) -> Result<(), Box<dyn std::error::Error>> {
    let provider = OpenAiCompatibleProvider::new(
        "local-openai-compatible",
        args.endpoint,
        vec![args.model.clone()],
    )
    .input_cost_per_1k(args.input_cost_per_1k)
    .output_cost_per_1k(args.output_cost_per_1k);
    let gateway = Gateway::builder(GatewayConfig::default())
        .cache(Arc::new(InMemoryCache::default()))
        .provider(Arc::new(provider))
        .build()?;
    let request = CompletionRequest::new(args.prompt).model(args.model);
    let first = gateway.complete(request.clone()).await?;
    let theme = Theme::detect();
    print_completion("LOCAL MODEL", &first, theme);
    if args.replay {
        let replay = gateway.complete(request).await?;
        print_completion("EXACT REPLAY", &replay, theme);
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct Theme {
    color: bool,
}
impl Theme {
    fn detect() -> Self {
        Self {
            color: std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none(),
        }
    }
    fn paint(self, code: &str, value: impl std::fmt::Display) -> String {
        if self.color {
            format!("\x1b[{code}m{value}\x1b[0m")
        } else {
            value.to_string()
        }
    }
    fn brand(self, value: impl std::fmt::Display) -> String {
        self.paint("1;36", value)
    }
    fn muted(self, value: impl std::fmt::Display) -> String {
        self.paint("2", value)
    }
    fn good(self, value: impl std::fmt::Display) -> String {
        self.paint("1;32", value)
    }
    fn warn(self, value: impl std::fmt::Display) -> String {
        self.paint("1;33", value)
    }
}

fn print_completion(label: &str, completion: &Completion, theme: Theme) {
    println!(
        "\n  {}",
        theme.brand(format!("── {label} ─────────────────────────────────────"))
    );
    let cache = completion.route.cache.to_uppercase();
    let cache_detail = completion
        .route
        .semantic_similarity
        .map(|score| format!(" (cosine {score:.3})"))
        .unwrap_or_default();
    let status = if completion.cached {
        theme.good(format!("✓ {cache}{cache_detail}"))
    } else {
        theme.warn(format!("• {cache}"))
    };
    println!("  Cache     {status}");
    if !completion.route.candidates.is_empty() {
        println!("  Routes");
        for (index, candidate) in completion.route.candidates.iter().enumerate() {
            println!(
                "    {}. {:<22} {:<28} in ${:.4}/1K · out ${:.4}/1K",
                index + 1,
                candidate.provider,
                candidate.model,
                candidate.input_cost_per_1k_usd,
                candidate.output_cost_per_1k_usd
            );
        }
    }
    for attempt in &completion.route.attempts {
        let marker = if attempt.outcome == "selected" {
            theme.good("✓")
        } else {
            theme.warn("↳")
        };
        println!(
            "  Attempt   {marker} {} → {}",
            attempt.provider, attempt.outcome
        );
    }
    println!("  Response  {}", completion.text);
    println!(
        "  Tokens    {} input · {} output",
        completion.usage.prompt_tokens, completion.usage.completion_tokens
    );
    println!("  Cost      ${:.6} this request", completion.cost.total_usd);
    if completion.cost.avoided_usd > 0.0 {
        println!(
            "  Saved     {}",
            theme.good(format!(
                "${:.6} via {} cache",
                completion.cost.avoided_usd,
                completion.cache_kind.as_deref().unwrap_or("unknown")
            ))
        );
    }
}
