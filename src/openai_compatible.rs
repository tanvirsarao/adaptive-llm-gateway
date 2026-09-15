//! Provider adapter for servers implementing OpenAI's chat-completions protocol.
//!
//! This is the bridge to self-hosted vLLM, llama.cpp and LM Studio instances,
//! as well as hosted OpenAI-compatible providers.
use crate::{CompletionRequest, LlmProvider, ProviderCompletion, ProviderError, Usage};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

pub struct OpenAiCompatibleProvider {
    name: String,
    base_url: String,
    api_key: Option<String>,
    models: Vec<String>,
    input_cost_per_1k: f64,
    output_cost_per_1k: f64,
    client: reqwest::Client,
}

impl OpenAiCompatibleProvider {
    /// `base_url` should include the API version, for example
    /// `http://127.0.0.1:8000/v1` for a local vLLM server.
    pub fn new(name: impl Into<String>, base_url: impl Into<String>, models: Vec<String>) -> Self {
        Self {
            name: name.into(),
            base_url: base_url.into().trim_end_matches('/').into(),
            api_key: None,
            models,
            input_cost_per_1k: 0.0,
            output_cost_per_1k: 0.0,
            client: reqwest::Client::new(),
        }
    }

    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set an estimated USD cost for a thousand input tokens. Use `0.0` for
    /// a local model so the router prioritizes it over paid providers.
    pub fn input_cost_per_1k(mut self, cost: f64) -> Self {
        self.input_cost_per_1k = cost;
        self
    }

    pub fn output_cost_per_1k(mut self, cost: f64) -> Self {
        self.output_cost_per_1k = cost;
        self
    }
}

#[async_trait]
impl LlmProvider for OpenAiCompatibleProvider {
    fn name(&self) -> &str {
        &self.name
    }
    fn models(&self) -> &[String] {
        &self.models
    }
    fn input_cost_per_1k(&self, _: &str) -> Option<f64> {
        Some(self.input_cost_per_1k)
    }
    fn output_cost_per_1k(&self, _: &str) -> Option<f64> {
        Some(self.output_cost_per_1k)
    }

    async fn complete(
        &self,
        request: &CompletionRequest,
    ) -> Result<ProviderCompletion, ProviderError> {
        let body = ChatRequest {
            model: request.model.clone().unwrap_or_else(|| {
                self.models
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "default".into())
            }),
            messages: vec![Message {
                role: "user",
                content: &request.prompt,
            }],
            max_tokens: request.max_tokens,
            temperature: request.temperature,
        };
        let mut call = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .json(&body);
        if let Some(api_key) = &self.api_key {
            call = call.bearer_auth(api_key);
        }
        let response = call.send().await.map_err(|error| ProviderError {
            message: error.to_string(),
            retryable: error.is_timeout() || error.is_connect(),
        })?;
        let status = response.status();
        if !status.is_success() {
            let message = response
                .text()
                .await
                .unwrap_or_else(|_| "unable to read provider error".into());
            return Err(ProviderError {
                message: format!("HTTP {status}: {message}"),
                retryable: status.as_u16() == 429 || status.is_server_error(),
            });
        }
        let parsed: ChatResponse = response.json().await.map_err(|error| ProviderError {
            message: format!("invalid provider response: {error}"),
            retryable: false,
        })?;
        let text = parsed
            .choices
            .into_iter()
            .next()
            .and_then(|choice| choice.message.content)
            .ok_or_else(|| ProviderError {
                message: "provider response had no text choice".into(),
                retryable: false,
            })?;
        Ok(ProviderCompletion {
            text,
            model: parsed.model,
            usage: parsed.usage.unwrap_or_default().into(),
        })
    }
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: String,
    messages: Vec<Message<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}
#[derive(Serialize)]
struct Message<'a> {
    role: &'static str,
    content: &'a str,
}
#[derive(Deserialize)]
struct ChatResponse {
    model: String,
    choices: Vec<Choice>,
    usage: Option<ResponseUsage>,
}
#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
}
#[derive(Deserialize)]
struct ResponseMessage {
    content: Option<String>,
}
#[derive(Default, Deserialize)]
struct ResponseUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}
impl From<ResponseUsage> for Usage {
    fn from(value: ResponseUsage) -> Self {
        Self {
            prompt_tokens: value.prompt_tokens,
            completion_tokens: value.completion_tokens,
        }
    }
}
