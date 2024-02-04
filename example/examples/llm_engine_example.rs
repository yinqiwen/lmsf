use clap::{Parser, ValueEnum};

use minfer_core::EngineArgs;
use minfer_core::LLMEngine;
use minfer_core::SamplingParams;

fn run(args: &EngineArgs) -> anyhow::Result<()> {
    let (model_cfg, cache_cfg, parallel_cfg, sched_cfg) = args.create_engine_configs()?;

    let mut engine = LLMEngine::from(model_cfg, cache_cfg, parallel_cfg, sched_cfg)?;

    let mut sampling_params = SamplingParams::default();
    sampling_params.temperature = 0.8;
    sampling_params.top_k = 5;
    sampling_params.presence_penalty = 0.2;
    sampling_params.max_tokens = 128;
    let max_tokens = sampling_params.max_tokens;

    let arrival_time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap();
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
    engine.add_request(
        request_id + 2,
        "To be or not to be, ",
        sampling_params.clone(),
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
                tracing::info!("{}:gen text:{}", req_out.request_id, out.text);
            }
        }
    }
    tracing::info!("End cost {:?} to gen {} tokens", start.elapsed(), count);

    // engine.add_request(
    //     request_id + 3,
    //     "To be or not to be, ",
    //     sampling_params,
    //     None,
    //     arrival_time,
    // )?;
    // let start = std::time::Instant::now();
    // while engine.has_unfinished_requests() {
    //     let outputs = engine.step()?;
    //     count += 1;
    //     // if count == 4 {
    //     //     break;
    //     // }
    //     for req_out in outputs {
    //         for out in req_out.outputs {
    //             tracing::info!("gen text:{}", out.text);
    //         }
    //     }
    // }
    // tracing::info!("End cost {:?} to gen {} tokens", start.elapsed(), count);
    Ok(())
}

fn main() {
    let args = EngineArgs::parse();
    tracing_subscriber::fmt::init();
    if let Err(e) = run(&args) {
        tracing::error!("{}", e);
    }
}
