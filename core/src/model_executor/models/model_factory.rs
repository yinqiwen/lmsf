use anyhow::anyhow;
use std::{
    fs::{read_to_string, File},
    // io::prelude::*,
    // io::BufReader,
    // ops::{Deref, DerefMut},
    path::{Path, PathBuf},
};

use crate::model_executor::layers::Cache;

use crate::model_executor::models::llama::{self, Llama, LlamaConfig, LlamaTokenizerConfig};
// use super::llama::{Config, Llama};
// use super::llama_ext::{LlamaConfig, LlamaModel, LlamaTokenizerConfig};
use super::{
    llama::LlamaChatTemplate,
    model::{Model, ModelConfig, TokenizerConfig},
    template::ChatTemplate,
};
pub struct ModelFactory;
impl ModelFactory {
    pub fn load_model(
        model_type: &str,
        cfg: &dyn ModelConfig,
        vb: candle_nn::VarBuilder,
    ) -> anyhow::Result<Box<dyn Model>> {
        match model_type {
            "llama" => {
                let dtype = cfg.get_dtype()?;
                let config = llama::Config::config_7b_v2(false);
                let cache = Cache::new(vb.device())?;
                let llama = Llama::load(vb, &cache, &config)?;
                // let llama = LlamaModel { llama };
                Ok(Box::new(llama))
            }
            _ => Err(anyhow!("invalid model_type:{}", model_type)),
        }
    }
    pub fn new_model_config(config_data: &str) -> anyhow::Result<Box<dyn ModelConfig>> {
        let cfg_value: serde_json::Value = serde_json::from_str(config_data)?;
        let cfg_dict = cfg_value.as_object().ok_or(anyhow!("invalid json"))?;
        let model_type_value = cfg_dict
            .get("model_type")
            .ok_or(anyhow!("no 'model_type' exist in config"))?;
        let model_type = model_type_value
            .as_str()
            .ok_or(anyhow!("invalid model_type value"))?;

        match model_type {
            "llama" => {
                let llama_cfg: LlamaConfig = serde_json::from_str(config_data)?;
                Ok(Box::new(llama_cfg))
            }
            _ => Err(anyhow!("invalid model_type:{}", model_type)),
        }
    }
    pub fn get_chat_template(
        model_type: &str,
        template: Option<&str>,
    ) -> anyhow::Result<Option<Box<dyn ChatTemplate>>> {
        if let Some(t) = template {
            match model_type {
                "llama" => match template {
                    Some(content) => {
                        let template = LlamaChatTemplate::new(content)?;
                        Ok(Some(Box::new(template)))
                    }
                    None => Ok(None),
                },
                _ => Err(anyhow!(
                    "invalid model_type:{} to get_chat_template",
                    model_type
                )),
            }
        } else {
            Ok(None)
        }
    }
    // pub fn from_file<P: AsRef<Path>>(file: P) -> Result<Self> {
    //     let content = read_to_string(file)?;
    //     let config = serde_json::from_str(&content)?;
    //     Ok(config)
    // }
    pub fn load_tokenizer_config<P: AsRef<Path>>(
        model_type: &str,
        file: P,
    ) -> anyhow::Result<Box<dyn TokenizerConfig>> {
        let content = read_to_string(file)?;
        match model_type {
            "llama" => {
                let cfg: LlamaTokenizerConfig = serde_json::from_str(content.as_str())?;
                Ok(Box::new(cfg))
            }
            _ => Err(anyhow!(
                "invalid model_type:{} to load_tokenizer_config",
                model_type
            )),
        }
    }
}
