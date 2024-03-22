use anyhow::anyhow;
use candle::Device;
use std::{
    collections::{HashMap, HashSet},
    str::FromStr,
};
use strum_macros::{Display, EnumString};

use super::{
    model::{Model, ModelConfig},
    template::ChatTemplate,
};
use crate::model_executor::layers::UnquantizedLinearWeights;
use crate::model_executor::models::llama::{Llama, LlamaConfig};
use crate::model_executor::{
    layers::{
        quantization::{AWQConfig, AWQLinearWeights, QuantizationConfig},
        Cache,
    },
    parallel::ParallelState,
};

use clap::ValueEnum;

#[derive(ValueEnum, Debug, Display, Clone, Copy, PartialEq, EnumString)]
pub enum QuantizeType {
    AWQ,
    // Q2_K,
    // Q3_K_L,
    // Q3_K_M,
    // Q3_K_S,
    // Q4_0,
    // Q4_K_M,
    // Q4_K_S,
    // Q5_0,
    // Q5_K_M,
    // Q5_K_S,
    // Q6_K,
    // Q8_0,
}

#[derive(ValueEnum, Debug, Clone, PartialEq, EnumString)]
#[strum(ascii_case_insensitive)]
pub enum ModelType {
    LLAMA,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
struct SafeTensorsIndex {
    weight_map: HashMap<String, String>,
}
pub struct ModelFactory;
impl ModelFactory {
    pub fn get_model_weight_files(
        model_path: &str,
        quantize_type: Option<QuantizeType>,
    ) -> anyhow::Result<Vec<String>> {
        let model_weight_files = match quantize_type {
            None => {
                let safetensors_index_path = format!("{}/model.safetensors.index.json", model_path);
                let safetensors_index_content = std::fs::read_to_string(safetensors_index_path)
                    .map_err(|e| anyhow!("No safetensors.index.json found!:{}", e))?;
                let safetensors_index: SafeTensorsIndex =
                    serde_json::from_str(safetensors_index_content.as_str())
                        .map_err(|e| anyhow!("Parse safetensors.index.json failed!:{}", e))?;

                let mut safetensors: HashSet<String> = HashSet::new();
                for (_, value) in safetensors_index.weight_map.iter() {
                    safetensors.insert(String::from(value));
                }
                let mut safetensors: Vec<_> = safetensors.into_iter().collect();
                safetensors.sort();
                safetensors
            }
            Some(QuantizeType::AWQ) => {
                vec!["model.safetensors".to_string()]
            } // Some(QuantizeType::Q2_K)
              // | Some(QuantizeType::Q3_K_L)
              // | Some(QuantizeType::Q3_K_M)
              // | Some(QuantizeType::Q3_K_S)
              // | Some(QuantizeType::Q4_0)
              // | Some(QuantizeType::Q4_K_M)
              // | Some(QuantizeType::Q4_K_S)
              // | Some(QuantizeType::Q5_0)
              // | Some(QuantizeType::Q5_K_M)
              // | Some(QuantizeType::Q5_K_S)
              // | Some(QuantizeType::Q6_K)
              // | Some(QuantizeType::Q8_0) => {
              //     let file_suffix = format!("{}.gguf", quantize_type.unwrap().to_string());
              //     let paths = std::fs::read_dir(model_path).unwrap();
              //     let mut files = Vec::new();
              //     for path in paths {
              //         match path {
              //             Ok(path) => {
              //                 let fname = path.file_name().into_string().unwrap();
              //                 if fname.ends_with(file_suffix.as_str()) {
              //                     files.push(format!("{}", fname));
              //                     break;
              //                 }
              //             }
              //             _ => {
              //                 continue;
              //             }
              //         };
              //     }
              //     files
              // }
        };
        Ok(model_weight_files)
    }

    pub fn load_model(
        dir: &str,
        quantize_type: Option<QuantizeType>,
        cfg: &dyn ModelConfig,
        model_weight_files: &Vec<String>,
        device: &Device,
    ) -> anyhow::Result<Box<dyn Model>> {
        let preinit_model = || -> anyhow::Result<_> {
            let is_pth = str::ends_with(model_weight_files[0].as_str(), ".pt");
            let is_safetensors = str::ends_with(model_weight_files[0].as_str(), ".safetensors");

            let vb = unsafe {
                if is_safetensors {
                    candle_nn::VarBuilder::from_mmaped_safetensors(
                        model_weight_files,
                        cfg.get_dtype()?,
                        &device,
                    )?
                } else if is_pth {
                    candle_nn::VarBuilder::from_pth(
                        model_weight_files[0].as_str(),
                        cfg.get_dtype()?,
                        &device,
                    )?
                } else {
                    panic!("Not supported model format:{:?}", model_weight_files)
                }
            };
            let _dtype = cfg.get_dtype()?;
            let cfg_any = cfg.as_any();
            let llama_cfg = match cfg_any.downcast_ref::<LlamaConfig>() {
                Some(b) => b,
                None => panic!("not LlamaConfig"),
            };
            let cache = Cache::new(vb.device())?;
            let parallel_state = ParallelState::default();
            Ok((vb, llama_cfg, cache, parallel_state))
        };

        match cfg.get_model_type() {
            ModelType::LLAMA => match quantize_type {
                None => {
                    let (vb, llama_cfg, cache, parallel_state) = preinit_model()?;
                    let llama = Llama::<UnquantizedLinearWeights>::load(
                        vb,
                        &cache,
                        llama_cfg,
                        &parallel_state,
                        None,
                    )?;
                    Ok(Box::new(llama))
                }
                Some(QuantizeType::AWQ) => {
                    let (vb, llama_cfg, cache, parallel_state) = preinit_model()?;
                    let awq_config = AWQConfig::load(dir)?;

                    let llama = Llama::<AWQLinearWeights>::load(
                        vb,
                        &cache,
                        llama_cfg,
                        &parallel_state,
                        awq_config,
                    )?;
                    Ok(Box::new(llama))
                } // Some(QuantizeType::Q2_K)
                  // | Some(QuantizeType::Q3_K_L)
                  // | Some(QuantizeType::Q3_K_M)
                  // | Some(QuantizeType::Q3_K_S)
                  // | Some(QuantizeType::Q4_0)
                  // | Some(QuantizeType::Q4_K_M)
                  // | Some(QuantizeType::Q4_K_S)
                  // | Some(QuantizeType::Q5_0)
                  // | Some(QuantizeType::Q5_K_M)
                  // | Some(QuantizeType::Q5_K_S)
                  // | Some(QuantizeType::Q6_K)
                  // | Some(QuantizeType::Q8_0) => {
                  //     let model_path = &model_weight_files[0];
                  //     let start = Instant::now();
                  //     let mut file = std::fs::File::open(model_path)?;
                  //     let model =
                  //         gguf_file::Content::read(&mut file).map_err(|e| e.with_path(model_path))?;
                  //     let mut total_size_in_bytes = 0;
                  //     for (_, tensor) in model.tensor_infos.iter() {
                  //         let elem_count = tensor.shape.elem_count();
                  //         total_size_in_bytes += elem_count * tensor.ggml_dtype.type_size()
                  //             / tensor.ggml_dtype.block_size();
                  //     }

                  //     let md_get = |s: &str| match model.metadata.get(s) {
                  //         None => Err(anyhow!("cannot find {s} in metadata")),
                  //         Some(v) => Ok(v),
                  //     };

                  //     let layers = md_get("llama.block_count")?.to_u32()? as usize;
                  //     let head_count = md_get("llama.attention.head_count")?.to_u32()? as usize;
                  //     let head_count_kv = md_get("llama.attention.head_count_kv")?.to_u32()? as usize;
                  //     let embedding_length = md_get("llama.embedding_length")?.to_u32()? as usize;
                  //     let rope_dim = md_get("llama.rope.dimension_count")?.to_u32()? as usize;
                  //     let rms_norm_eps =
                  //         md_get("llama.attention.layer_norm_rms_epsilon")?.to_f32()?;
                  //     cfg.set_num_hidden_layers(layers);
                  //     cfg.set_num_attention_heads(head_count);
                  //     cfg.set_num_key_value_heads(head_count_kv);
                  //     cfg.set_hidden_size(embedding_length);

                  //     tracing::info!(
                  //         "loaded {:?} tensors, embedding_length:{},rms_norm_eps:{},rope_dim:{},in {:?} ",
                  //         model.tensor_infos.len(),
                  //         embedding_length,
                  //         rms_norm_eps,
                  //         rope_dim,
                  //         start.elapsed(),
                  //     );
                  //     let model =
                  //         quantized_llama::ModelWeights::from_gguf(model, &mut file, &device)?;
                  //     Ok(Box::new(model))
                  // }
            },
        }
    }
    pub fn new_model_config(
        model_path: &str,
        quantize_type: Option<QuantizeType>,
    ) -> anyhow::Result<Box<dyn ModelConfig>> {
        let config_path = format!("{}/config.json", model_path);
        let config_data = std::fs::read_to_string(config_path)
            .expect("Should have been able to read the config file");

        let cfg_value: serde_json::Value = serde_json::from_str(&config_data)?;
        let cfg_dict = cfg_value.as_object().ok_or(anyhow!("invalid json"))?;
        let model_type_value = cfg_dict
            .get("model_type")
            .ok_or(anyhow!("no 'model_type' exist in config"))?;
        let model_type = model_type_value
            .as_str()
            .ok_or(anyhow!("invalid model_type value"))?;

        let model_type = <ModelType as FromStr>::from_str(model_type).map_err(|r| {
            anyhow!(
                "create model type from:{} failed with reason:{}",
                model_type,
                r
            )
        })?;

        match model_type {
            ModelType::LLAMA => match quantize_type {
                None | Some(QuantizeType::AWQ) => {
                    let llama_cfg: LlamaConfig =
                        serde_json::from_str(&config_data).map_err(|e| {
                            tracing::error!("Failed to parse LlamaConfig with error:{:?}", e);
                            e
                        })?;
                    Ok(Box::new(llama_cfg))
                } // Some(QuantizeType::Q2_K)
                  // | Some(QuantizeType::Q3_K_L)
                  // | Some(QuantizeType::Q3_K_M)
                  // | Some(QuantizeType::Q3_K_S)
                  // | Some(QuantizeType::Q4_0)
                  // | Some(QuantizeType::Q4_K_M)
                  // | Some(QuantizeType::Q4_K_S)
                  // | Some(QuantizeType::Q5_0)
                  // | Some(QuantizeType::Q5_K_M)
                  // | Some(QuantizeType::Q5_K_S)
                  // | Some(QuantizeType::Q6_K)
                  // | Some(QuantizeType::Q8_0) => {
                  //     let llama_cfg = LlamaConfig::default();
                  //     Ok(Box::new(llama_cfg))
                  // }
            },
        }
    }
    pub fn get_chat_template(model_path: &str) -> anyhow::Result<Option<ChatTemplate>> {
        let tokenizer_config_path = format!("{}/tokenizer_config.json", model_path);
        if let Ok(config_data) = std::fs::read_to_string(tokenizer_config_path) {
            let cfg_value: serde_json::Value = serde_json::from_str(&config_data)?;
            let cfg_dict = cfg_value.as_object().ok_or(anyhow!("invalid json"))?;
            if let Some(chat_template) = cfg_dict.get("chat_template") {
                let chat_template = chat_template
                    .as_str()
                    .ok_or(anyhow!("invalid chat_template value"))?;
                let chat_template = ChatTemplate::new(chat_template)?;
                Ok(Some(chat_template))
            } else {
                Ok(None)
            }
        } else {
            let chat_template = format!("{}/chat_template", model_path);
            if let Ok(content) = std::fs::read_to_string(chat_template) {
                let chat_template = ChatTemplate::new(content.as_str())?;
                Ok(Some(chat_template))
            } else {
                Ok(None)
            }
        }
    }

    // pub fn load_tokenizer_config<P: AsRef<Path>>(
    //     model_type: ModelType,
    //     file: P,
    // ) -> anyhow::Result<Box<dyn TokenizerConfig>> {
    //     let content = read_to_string(file)?;
    //     match model_type {
    //         ModelType::LLAMA => {
    //             let cfg: LlamaTokenizerConfig = serde_json::from_str(content.as_str())?;
    //             Ok(Box::new(cfg))
    //         }
    //         _ => Err(anyhow!(
    //             "invalid model_type:{:?} to load_tokenizer_config",
    //             model_type
    //         )),
    //     }
    // }
}
