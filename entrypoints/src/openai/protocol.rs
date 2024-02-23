use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;

#[derive(Serialize)]
pub enum ErrorType {
    Unhealthy,
    Backend,
    Overloaded,
    Validation,
    Tokenizer,
}

#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub error_type: ErrorType,
}

fn get_unix_secs() -> u32 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs() as u32
}

#[derive(Serialize)]
pub struct ModelPermission {
    pub id: String,
    pub object: String,
    pub created: u32,
    pub allow_create_engine: bool,
    pub allow_sampling: bool,
    pub allow_logprobs: bool,
    pub allow_search_indices: bool,
    pub allow_view: bool,
    pub allow_fine_tuning: bool,
    pub organization: String,
    pub group: Option<String>,
    pub is_blocking: bool,
}

impl Default for ModelPermission {
    fn default() -> Self {
        Self {
            id: format!("modelperm-{}", uuid::Uuid::new_v4()),
            object: String::from("model_permission"),
            created: get_unix_secs(),
            allow_create_engine: false,
            allow_sampling: true,
            allow_logprobs: true,
            allow_search_indices: false,
            allow_view: true,
            allow_fine_tuning: false,
            organization: String::from("*"),
            group: None,
            is_blocking: false,
        }
    }
}

#[derive(Serialize)]
pub struct ModelCard {
    pub id: String,
    pub object: String,
    pub created: u32,
    pub owned_by: String,
    pub root: Option<String>,
    pub parent: Option<String>,
    pub permission: Vec<ModelPermission>,
}

impl Default for ModelCard {
    fn default() -> Self {
        Self {
            id: Default::default(),
            object: String::from("model"),
            created: get_unix_secs(),
            owned_by: String::from("lmsf"),
            root: None,
            parent: None,
            permission: vec![ModelPermission::default()],
        }
    }
}

#[derive(Serialize)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelCard>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub typ: String,
}
#[derive(Deserialize, Serialize, Debug)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub name: Option<String>,
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    pub fn from(role: &str, content: &str) -> Self {
        Self {
            role: String::from(role),
            content: String::from(content),
            name: None,
            tool_call_id: None,
        }
    }
    pub fn to_dict(&self) -> HashMap<String, String> {
        let mut h = HashMap::new();
        h.insert("role".to_string(), self.role.clone());
        h.insert("content".to_string(), self.content.clone());
        h
    }
}

#[derive(Deserialize, Serialize, Debug)]
pub struct ChatCompletionRequest {
    pub messages: Vec<ChatMessage>,
    pub model: String,
    pub frequency_penalty: Option<f32>,
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<i32>,
    pub max_tokens: Option<i32>,
    pub n: Option<usize>,
    pub presence_penalty: Option<f32>,
    pub response_format: Option<ResponseFormat>,
    pub seed: Option<i32>,
    pub stop: Option<(String, Vec<String>)>,
    pub stream: Option<bool>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub logit_bias: Option<HashMap<String, f32>>,
    pub user: Option<String>,
    // Additional parameters supported
    pub best_of: Option<i32>,
    pub top_k: Option<i32>,
    pub ignore_eos: Option<bool>,
    pub use_beam_search: Option<bool>,
    pub stop_token_ids: Option<Vec<i32>>,
    pub skip_special_tokens: Option<bool>,
    pub spaces_between_special_tokens: Option<bool>,
    pub add_generation_prompt: Option<bool>,
    pub echo: Option<bool>,
    pub repetition_penalty: Option<f32>,
    pub min_p: Option<f32>,
}
#[derive(Deserialize, Serialize, Debug)]
pub struct ChatCompletionResponseChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}
#[derive(Deserialize, Serialize, Debug)]
pub struct UsageInfo {
    pub completion_tokens: usize,
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}
#[derive(Deserialize, Serialize, Debug)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i32,
    pub model: String,
    pub choices: Vec<ChatCompletionResponseChoice>,
    pub usage: UsageInfo,
    // pub system_fingerprint: String,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct DeltaMessage {
    pub role: Option<String>,
    pub content: Option<String>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct ChatCompletionResponseStreamChoice {
    pub index: usize,
    pub delta: DeltaMessage,
    pub finish_reason: Option<String>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct ChatCompletionStreamResponse {
    pub id: String,
    pub object: String,
    pub created: u32,
    pub model: String,
    pub choices: Vec<ChatCompletionResponseStreamChoice>,
    pub usage: Option<UsageInfo>,
}
