use std::{io::Write, ops::Sub, time::Duration};

use clap::{Parser, ValueEnum};

use lmsf_core::{
    AsyncLLMEngine, EngineArgs, LLMEngine, LLMPrompt, LLMTaskResponseReceiver, SamplingParams,
};

async fn async_run(args: &EngineArgs) -> anyhow::Result<()> {
    let (model_cfg, cache_cfg, parallel_cfg, sched_cfg) = args.create_engine_configs()?;
    let runner = AsyncLLMEngine::new(model_cfg, cache_cfg, parallel_cfg, sched_cfg, true).await;

    let mut sampling_params = SamplingParams::default();
    sampling_params.temperature = 0.8;
    sampling_params.top_k = 5;
    sampling_params.presence_penalty = 0.2;
    sampling_params.max_tokens = 128;

    for i in 0..3 {
        let prompt = "To be or not to be,".to_string();
        let prompt = LLMPrompt::from(prompt);
        let start = std::time::Instant::now();
        let receiver = runner.add(prompt, sampling_params.clone(), true)?;
        match receiver {
            LLMTaskResponseReceiver::Normal(rx) => {
                if let Ok(output) = rx.await {
                    for output in output.outputs {
                        //tracing::info!("gen text:{}", output.text);
                        print!("{}", output.latest_token);
                        std::io::stdout().flush();
                    }
                }
            }
            LLMTaskResponseReceiver::Stream(mut rx) => {
                tracing::info!("stream");
                loop {
                    match rx.recv().await {
                        Some(result) => {
                            for output in result.outputs {
                                print!("{}", output.latest_token);
                                std::io::stdout().flush();
                            }
                        }
                        None => {
                            break;
                        }
                    }
                }
            }
        }

        println!("\n[{}]cost {:?}!", i, start.elapsed());
    }

    Ok(())
}

fn run(args: &EngineArgs) -> anyhow::Result<()> {
    let (model_cfg, cache_cfg, parallel_cfg, sched_cfg) = args.create_engine_configs()?;

    let mut engine = LLMEngine::from(model_cfg, cache_cfg, parallel_cfg, sched_cfg)?;

    let mut sampling_params = SamplingParams::default();
    sampling_params.temperature = 0.8;
    sampling_params.top_k = 5;
    sampling_params.presence_penalty = 0.2;
    sampling_params.max_tokens = 128;
    let max_tokens = sampling_params.max_tokens;

    let arrival_time = std::time::Instant::now();
    let request_id = 0_u64;
    // engine.add_request(
    //     request_id,
    //     "To be or not to be, ",
    //     sampling_params.clone(),
    //     None,
    //     arrival_time,
    // )?;
    // //engine.add_request(request_id, prompt, sampling_params, arrival_time)

    // while engine.has_unfinished_requests() {
    //     let _ = engine.step()?;
    // }
    let mut count = 0;
    tracing::info!("start");
    engine.add_request(
        request_id + 1,
        "To be or not to be, ",
        sampling_params.clone(),
        None,
        arrival_time,
    )?;
    // engine.add_request(
    //     request_id + 2,
    //     "To be or not to be, ",
    //     sampling_params.clone(),
    //     None,
    //     arrival_time,
    // )?;
    // engine.add_request(
    //     request_id + 3,
    //     "To be or not to be, ",
    //     sampling_params.clone(),
    //     None,
    //     arrival_time,
    // )?;
    let start = std::time::Instant::now();
    while engine.has_unfinished_requests() {
        let outputs = engine.step()?;
        count += 1;
        // if count == 4 {
        //     break;
        // }
        for req_out in outputs {
            for out in req_out.outputs {
                //tracing::info!("{}:gen text:{}", req_out.request_id, out.text);
            }
        }
    }
    tracing::info!("End cost {:?} to gen {} tokens", start.elapsed(), count);

    engine.add_request(
        request_id + 3,
        "To be or not to be, ",
        sampling_params,
        None,
        arrival_time,
    )?;
    let start = std::time::Instant::now();
    while engine.has_unfinished_requests() {
        let outputs = engine.step()?;
        count += 1;
        // if count == 4 {
        //     break;
        // }
        for req_out in outputs {
            for out in req_out.outputs {
                //tracing::info!("gen text:{}", out.text);
            }
        }
    }
    tracing::info!("End cost {:?} to gen {} tokens", start.elapsed(), count);
    Ok(())
}

#[tokio::main]
async fn main() {
    let args = EngineArgs::parse();

    //tracing_subscriber::fmt::init();
    common::MetricsBuilder::new().install();
    common::init_tracing(None, true);

    let current_dir = std::env::current_dir().unwrap();
    tracing::info!("Working dir:{:?}:", current_dir);
    // if let Err(e) = run(&args) {
    //     tracing::error!("{}", e);
    // }
    if let Err(e) = async_run(&args).await {
        tracing::error!("{}", e);
    }
    tokio::time::sleep(Duration::from_secs(70)).await;
}
