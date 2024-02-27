use axum::{
    extract::Extension,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use clap::{Parser, ValueEnum};
use lmsf_core::{AsyncLLMEngine, EngineArgs};
use serde::{Deserialize, Serialize};

mod openai;

#[tokio::main]
async fn main() {
    let args = EngineArgs::parse();
    tracing_subscriber::fmt::init();

    let (model_cfg, cache_cfg, parallel_cfg, sched_cfg) = match args.create_engine_configs() {
        Ok((model_cfg, cache_cfg, parallel_cfg, sched_cfg)) => {
            (model_cfg, cache_cfg, parallel_cfg, sched_cfg)
        }
        Err(e) => {
            tracing::info!("failed to parse args:{}", e);
            return;
        }
    };
    // let (model_cfg, cache_cfg, parallel_cfg, sched_cfg) = args.create_engine_configs()?;
    let engine = AsyncLLMEngine::new(model_cfg, cache_cfg, parallel_cfg, sched_cfg).await;

    // build our application with a route
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(openai::show_available_models))
        // `POST /users` goes to `create_user`
        .route("/v1/chat/completions", post(openai::create_chat_completion));

    let app = app.layer(Extension(engine));
    // run our app with hyper, listening globally on port 3000
    let addr = format!("{}:{}", args.host, args.port);
    tracing::info!("Start API Server at address:{}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn health() -> &'static str {
    ""
}
