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
        // `GET /` goes to `root`
        .route("/", get(root))
        .route("/health", get(health))
        .route("/v1/models", get(openai::show_available_models))
        // `POST /users` goes to `create_user`
        .route("/v1/chat/completions", post(openai::create_chat_completion));

    let app = app.layer(Extension(engine));
    // run our app with hyper, listening globally on port 3000
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn health() -> &'static str {
    ""
}

// basic handler that responds with a static string
async fn root() -> &'static str {
    "Hello, World!"
}

// async fn create_user(
//     // this argument tells axum to parse the request body
//     // as JSON into a `CreateUser` type
//     Json(payload): Json<CreateUser>,
// ) -> (StatusCode, Json<User>) {
//     // insert your application logic here
//     let user = User {
//         id: 1337,
//         username: payload.username,
//     };

//     // this will be converted into a JSON response
//     // with a status code of `201 Created`
//     (StatusCode::CREATED, Json(user))
// }

// // the input to our `create_user` handler
// #[derive(Deserialize)]
// struct CreateUser {
//     username: String,
// }

// // the output to our `create_user` handler
// #[derive(Serialize)]
// struct User {
//     id: u64,
//     username: String,
// }
