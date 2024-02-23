use axum::{
    body::Body,
    extract::Extension,
    http::{header, request, HeaderMap, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use futures_core::stream::Stream;
use tokio::sync::mpsc::{self, error::TryRecvError};
// use futures_util::pin_mut;
// use futures_util::stream::StreamExt;

use lmsf_core::{
    AsyncLLMEngine, LLMPrompt, LLMTaskResponseReceiver, RequestOutput, SamplingParams,
};

use crate::openai::protocol::{ChatCompletionResponseChoice, ChatMessage, UsageInfo};

use super::protocol::{
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse, DeltaMessage, ErrorResponse, ModelCard, ModelList,
};

pub async fn show_available_models(
    engine: Extension<AsyncLLMEngine>,
) -> Result<(HeaderMap, Json<ModelList>), (StatusCode, Json<ErrorResponse>)> {
    let headers = HeaderMap::new();

    let mut model = ModelCard::default();
    model.id = String::from(engine.model());
    let model_list = ModelList {
        object: String::from("list"),
        data: vec![model],
    };
    Ok((headers, Json(model_list)))
}

fn get_full_completion(
    request_id: &str,
    model: &str,
    role: &str,
    final_res: RequestOutput,
) -> impl Stream<Item = anyhow::Result<String>> {
    let mut choices: Vec<ChatCompletionResponseChoice> = Vec::new();
    let mut num_generated_tokens: usize = 0;
    for out in final_res.outputs {
        num_generated_tokens += out.token_ids.len();
        let choice = ChatCompletionResponseChoice {
            index: out.index,
            message: ChatMessage::from(role, out.text.as_str()),
            finish_reason: out.get_finish_reason(),
        };
        choices.push(choice);
    }
    let num_prompt_tokens = final_res.prompt_token_ids.len();
    let usage = UsageInfo {
        prompt_tokens: num_generated_tokens,
        completion_tokens: num_generated_tokens,
        total_tokens: num_prompt_tokens + num_generated_tokens,
    };
    let created_time = 0_i32;
    let response = ChatCompletionResponse {
        id: String::from(request_id),
        object: String::from("chat.completion"),
        created: created_time,
        model: String::from(model),
        choices,
        usage,
    };
    async_stream::stream! {
        yield Ok(serde_json::to_string(&response).unwrap());
    }
}

fn get_stream_completion(
    n: usize,
    request_id: String,
    model: String,
    role: String,
    mut receiver: mpsc::Receiver<RequestOutput>,
) -> impl Stream<Item = anyhow::Result<String>> {
    // previous_texts = [""] * request.n
    // previous_num_tokens = [0] * request.n
    // finish_reason_sent = [False] * request.n
    let mut previous_num_tokens = [0_usize].repeat(n);
    let mut finish_reason_sent = [false].repeat(n);
    async_stream::stream! {
        for i in 0..n{
            let choice_data = ChatCompletionResponseStreamChoice{
                index:i,
                delta:DeltaMessage{
                    role:Some(role.clone()),
                    content:None,
                },
                finish_reason:None,
            };
            let chunk = ChatCompletionStreamResponse{
                id: request_id.clone(),
                object:String::from("chat.completion.chunk"),
                created:0,
                model:model.clone(),
                choices:vec![choice_data],
                usage:None,
            };
            match serde_json::to_string(&chunk){
                Ok(data)=> yield Ok(format!("data: {}\n\n", data)),
                Err(e)=> yield Err(anyhow::anyhow!("{}", e)),
            }
        }
        loop{
            match receiver.recv().await{
                Some(result)=>{
                    for output in result.outputs{
                        let i = output.index;
                        if finish_reason_sent[i]{
                            continue;
                        }
                        previous_num_tokens[i] = output.token_ids.len();
                        let choice_data = ChatCompletionResponseStreamChoice{
                            index:i,
                            delta:DeltaMessage{
                                role:None,
                                content:Some(output.latest_token.clone()),
                            },
                            finish_reason:output.finish_reason.map(|s| s.to_string()),
                        };
                        let usage = match output.finish_reason{
                            Some(reason)=>{
                                finish_reason_sent[i] = true;
                                let prompt_tokens = result.prompt_token_ids.len();
                                Some(UsageInfo{
                                    prompt_tokens,
                                    completion_tokens:previous_num_tokens[i],
                                    total_tokens:prompt_tokens + previous_num_tokens[i],
                                })
                            },
                            None=>{
                                None
                            }
                        };
                        let chunk = ChatCompletionStreamResponse{
                            id:request_id.clone(),
                            object:String::from("chat.completion.chunk"),
                            created:0,
                            model:model.clone(),
                            choices:vec![choice_data],
                            usage,
                        };
                        match serde_json::to_string(&chunk){
                            Ok(data)=> yield Ok(format!("data: {}\n\n", data)),
                            Err(e)=> yield Err(anyhow::anyhow!("{}", e)),
                        }
                    }
                }
                None=>{
                    // println!("#####recv none!");
                    break;
                }
            }
        }
        yield Ok("data: [DONE]\n\n".to_string());
    }
}

pub async fn create_chat_completion(
    engine: Extension<AsyncLLMEngine>,
    Json(request): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    if let Some(logit_bias) = request.logit_bias {
        if logit_bias.len() > 0 {
            return Err((
                StatusCode::BAD_REQUEST,
                "logit_bias is not currently supported",
            ));
        }
    }

    let last_msg_idx = request.messages.len() - 1;
    let role = request.messages[last_msg_idx].role.as_str();
    let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());
    let model = request.model.as_str();
    let n = if let Some(n) = request.n { n } else { 1 };

    let mut messages = Vec::new();
    for m in &request.messages {
        messages.push(m.to_dict());
    }
    let stream = if let Some(stream) = request.stream {
        stream
    } else {
        false
    };
    let sampling_params = SamplingParams::default();
    let prompt = LLMPrompt::multi_role(messages);
    let output = match engine.add(prompt, sampling_params, stream) {
        Ok(output) => output,
        Err(e) => {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                "add prompt into engine failed",
            ));
        }
    };

    match output {
        LLMTaskResponseReceiver::Normal(o) => match o.await {
            Ok(result) => {
                let response = get_full_completion(request_id.as_str(), model, role, result);
                let headers = [(header::CONTENT_TYPE, "application/json")];
                let body = Body::from_stream(response);
                Ok((headers, body))
            }
            Err(e) => Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                "recv prompt response from engine failed",
            )),
        },
        LLMTaskResponseReceiver::Stream(mut o) => {
            let headers = [(header::CONTENT_TYPE, "text/event-stream")];
            println!("#####stream!");
            let body = Body::from_stream(get_stream_completion(
                n,
                request_id.clone(),
                String::from(model),
                String::from(role),
                o,
            ));
            Ok((headers, body))
        }
    }
    //engine.add(prompt, sampling_params, stream)

    // todo!("create_chat_completion")
}
