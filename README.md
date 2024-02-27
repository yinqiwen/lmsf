Rust LLM Serving Framework

## Features

-  Paged Attention
-  Continuous Batch


# Getting Started

**Examples**
```sh
$ cargo run --release --example llm_engine_example -- --model <llma model dir> --gpu-memory-utilization 0.95 --block-size 8 --max-model-len 1024
```

**API Server**
```sh
$ cargo build --release
$ ./target/release/entrypoints --model <llma model dir> --gpu-memory-utilization 0.95 --block-size 8 --max-model-len 1024 --host 0.0.0.0 --port 8000
```




