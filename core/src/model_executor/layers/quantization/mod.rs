use candle_core::DType;

mod awq;
pub trait QuantizationConfig: Sized {
    fn get_name() -> &'static str;
    fn get_supported_act_dtypes() -> Vec<DType>;
    fn get_config_filenames() -> Vec<&'static str>;
    fn get_min_capability() -> u32;
    fn get_scaled_act_names() -> Vec<&'static str>;

    fn from_json(value: serde_json::Value) -> candle_core::Result<Self>;

    fn load(dir: &str) -> candle_core::Result<Self> {
        for config_file in Self::get_config_filenames() {
            let config_path = format!("{}/{}", dir, config_file);
            let path = std::path::Path::new(config_path.as_str());
            if !path.exists() {
                continue;
            }
            let config_data =
                std::fs::read_to_string(config_path).map_err(|e| candle_core::Error::msg(e))?;
            let cfg_value: serde_json::Value =
                serde_json::from_str(&config_data).map_err(|e| candle_core::Error::msg(e))?;
            return Self::from_json(cfg_value);
        }
        candle_core::bail!("no valid config found")
    }
}

pub use awq::{AWQConfig, AWQLinearWeights};

// @classmethod
// @abstractmethod
// def from_config(cls, config: Dict[str, Any]) -> "QuantizationConfig":
//     """Create a config class from the model's quantization config."""
//     raise NotImplementedError

// @staticmethod
// def get_from_keys(config: Dict[str, Any], keys: List[str]) -> Any:
//     """Get a value from the model's quantization config."""
//     for key in keys:
//         if key in config:
//             return config[key]
//     raise ValueError(f"Cannot find any of {keys} in the model's "
//                      "quantization config.")

// @abstractmethod
// def get_linear_method(self) -> LinearMethodBase:
//     """Get the linear method to use for the quantized linear layer."""
//     raise NotImplementedError
