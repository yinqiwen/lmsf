use anyhow::anyhow;

#[derive(Debug, Eq, PartialEq, Hash)]
pub enum SamplingType {
    Greedy = 0,
    Random,
    Beam,
}

impl SamplingType {
    pub fn from_int(value: usize) -> Option<Self> {
        match value {
            0 => Some(SamplingType::Greedy),
            1 => Some(SamplingType::Random),
            2 => Some(SamplingType::Beam),
            _ => None,
        }
    }
}

const _SAMPLING_EPS: f32 = 1e-5;
#[derive(PartialEq, Clone, Debug)]
pub enum EarlyStopType {
    Never,
    EarlyStop(bool),
}

#[derive(PartialEq, Clone, Debug)]
pub struct SamplingParams {
    pub n: usize,
    pub best_of: Option<usize>,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub repetition_penalty: f32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub min_p: f32,
    pub use_beam_search: bool,
    pub length_penalty: f32,
    pub early_stopping: EarlyStopType,
    pub stop: Vec<String>,
    pub stop_token_ids: Option<Vec<u32>>,
    pub include_stop_str_in_output: bool,
    pub ignore_eos: bool,
    pub max_tokens: usize,
    pub logprobs: Option<u32>,
    pub prompt_logprobs: Option<u32>,
    pub skip_special_tokens: bool,
    pub spaces_between_special_tokens: bool,
    //logits_processors: Optional[List[LogitsProcessor]] = None,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            n: 1,
            best_of: Some(1),
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            temperature: 1.0,
            top_p: 1.0,
            top_k: -1,
            min_p: 0.0,
            use_beam_search: false,
            length_penalty: 1.0,
            early_stopping: EarlyStopType::EarlyStop(false),
            stop: Vec::new(),
            stop_token_ids: None,
            include_stop_str_in_output: false,
            ignore_eos: false,
            max_tokens: 16,
            logprobs: None,
            prompt_logprobs: None,
            skip_special_tokens: true,
            spaces_between_special_tokens: true,
        }
    }
}

impl SamplingParams {
    pub fn sampling_type(&self) -> SamplingType {
        if self.use_beam_search {
            SamplingType::Beam
        } else if self.temperature < _SAMPLING_EPS {
            SamplingType::Greedy
        } else {
            SamplingType::Random
        }
    }

    pub fn verify(&mut self) -> anyhow::Result<()> {
        if self.n < 1 {
            return Err(anyhow!("n must be at least 1, got {}.", self.n));
        }
        if let Some(best_of) = self.best_of {
            if best_of < self.n {
                return Err(anyhow!(
                    "best_of must be greater than or equal to n, got n={} and best_of={}.",
                    self.n,
                    best_of
                ));
            }
        }
        if self.presence_penalty < -2.0 || self.presence_penalty > 2.0 {
            return Err(anyhow!(
                "presence_penalty must be in [-2, 2], got {}.",
                self.presence_penalty
            ));
        }
        if self.frequency_penalty < -2.0 || self.frequency_penalty > 2.0 {
            return Err(anyhow!(
                "frequency_penalty must be in [-2, 2], got {}.",
                self.frequency_penalty
            ));
        }
        if self.repetition_penalty < 0.0 || self.repetition_penalty > 2.0 {
            return Err(anyhow!(
                "repetition_penalty must be in [0, 2], got {}.",
                self.repetition_penalty
            ));
        }
        if self.temperature < 0.0 {
            return Err(anyhow!(
                "temperature must be non-negative, got {}.",
                self.temperature
            ));
        }
        if self.top_p < 0.0 || self.top_p > 1.0 {
            return Err(anyhow!("top_p must be in (0, 1], got {}.", self.top_p));
        }
        if self.top_k < -1 || self.top_k == 0 {
            return Err(anyhow!(
                "top_k must be -1 (disable), or at least 1, got {}.",
                self.top_k
            ));
        }
        if self.min_p < 0.0 || self.min_p > 1.0 {
            return Err(anyhow!("min_p must be in [0, 1], got {}.", self.min_p));
        }
        if self.max_tokens < 1 {
            return Err(anyhow!(
                "max_tokens must be at least 1, got {}.",
                self.max_tokens
            ));
        }
        // if let Some(logprobs) = self.logprobs {
        //     if logprobs < 0 {
        //         return Err(anyhow!("logprobs must be non-negative, got {}.", logprobs));
        //     }
        // }
        // if let Some(prompt_logprobs) = self.prompt_logprobs {
        //     if prompt_logprobs < 0 {
        //         return Err(anyhow!(
        //             "prompt_logprobs must be non-negative, got {}.",
        //             prompt_logprobs
        //         ));
        //     }
        // }
        if self.use_beam_search {
            self.verify_beam_search()
        } else {
            self.verify_non_beam_search()?;
            if self.temperature < _SAMPLING_EPS {
                self.top_p = 1.0;
                self.top_k = -1;
                self.min_p = 0.0;
                self.verify_greedy_sampling()
            } else {
                Ok(())
            }
        }
    }

    fn verify_beam_search(&self) -> anyhow::Result<()> {
        if let Some(best_of) = self.best_of {
            if best_of == 1 {
                return Err(anyhow!(
                    "best_of must be greater than 1 when using beam search. Got {}.",
                    best_of
                ));
            }
        }
        if self.temperature > _SAMPLING_EPS {
            return Err(anyhow!("temperature must be 0 when using beam search."));
        }
        if self.top_p > 1.0 - _SAMPLING_EPS {
            return Err(anyhow!("top_p must be 1 when using beam search."));
        }
        if self.top_k != -1 {
            return Err(anyhow!("top_k must be -1 when using beam search."));
        }

        Ok(())
    }
    fn verify_non_beam_search(&self) -> anyhow::Result<()> {
        match self.early_stopping {
            EarlyStopType::EarlyStop(false) => {
                //none
            }
            _ => {
                return Err(anyhow!(
                    "early_stopping is not effective and must be False when not using beam search"
                ));
            }
        }

        if self.length_penalty < 1.0 - _SAMPLING_EPS || self.length_penalty > 1.0 + _SAMPLING_EPS {
            return Err(anyhow!("length_penalty is not effective and must be the default value of 1.0 when not using beam search."));
        }
        Ok(())
    }

    fn verify_greedy_sampling(&self) -> anyhow::Result<()> {
        if let Some(best_of) = self.best_of {
            if best_of > 1 {
                return Err(anyhow!(
                    "best_of must be 1 when using greedy sampling. Got {}.",
                    best_of
                ));
            }
        }
        Ok(())
    }
}

pub struct SamplingParamsBuilder {
    pub n: Option<usize>,
    pub best_of: Option<usize>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub min_p: Option<f32>,
    pub use_beam_search: Option<bool>,
    pub length_penalty: Option<f32>,
    pub early_stopping: Option<EarlyStopType>,
    pub stop: Option<Vec<String>>,
    pub stop_token_ids: Option<Vec<u32>>,
    pub include_stop_str_in_output: Option<bool>,
    pub ignore_eos: Option<bool>,
    pub max_tokens: Option<usize>,
    pub logprobs: Option<u32>,
    pub prompt_logprobs: Option<u32>,
    pub skip_special_tokens: Option<bool>,
    pub spaces_between_special_tokens: Option<bool>,
}

impl SamplingParamsBuilder {
    pub fn new() -> Self {
        Self {
            n: None,
            best_of: None,
            presence_penalty: None,
            frequency_penalty: None,
            repetition_penalty: None,
            temperature: None,
            top_p: None,
            top_k: None,
            min_p: None,
            use_beam_search: None,
            length_penalty: None,
            early_stopping: None,
            stop: None,
            stop_token_ids: None,
            include_stop_str_in_output: None,
            ignore_eos: None,
            max_tokens: None,
            logprobs: None,
            prompt_logprobs: None,
            skip_special_tokens: None,
            spaces_between_special_tokens: None,
        }
    }

    pub fn build() -> anyhow::Result<SamplingParams> {
        let params = SamplingParams::default();

        Ok(params)
    }
}
