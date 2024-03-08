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
            max_tokens: 128,
            logprobs: None,
            prompt_logprobs: None,
            skip_special_tokens: true,
            spaces_between_special_tokens: true,
        }
    }
}

impl SamplingParams {
    pub fn new() -> Self {
        SamplingParams::default()
    }
    pub fn with_n(mut self, n: usize) -> Self {
        self.n = n;
        self
    }
    pub fn with_best_of(mut self, best_of: usize) -> Self {
        self.best_of = Some(best_of);
        self
    }
    pub fn with_frequency_penalty(mut self, frequency_penalty: f32) -> Self {
        self.frequency_penalty = frequency_penalty;
        self
    }
    pub fn with_repetition_penalty(mut self, repetition_penalty: f32) -> Self {
        self.repetition_penalty = repetition_penalty;
        self
    }

    pub fn with_min_p(mut self, min_p: f32) -> Self {
        self.min_p = min_p;
        self
    }

    pub fn with_beam_search(mut self) -> Self {
        self.use_beam_search = true;
        self
    }
    pub fn with_length_penalty(mut self, length_penalty: f32) -> Self {
        self.length_penalty = length_penalty;
        self
    }
    pub fn with_ignore_eos(mut self) -> Self {
        self.ignore_eos = true;
        self
    }
    pub fn disable_skip_special_tokens(mut self) -> Self {
        self.skip_special_tokens = false;
        self
    }
    pub fn disable_spaces_between_special_tokens(mut self) -> Self {
        self.spaces_between_special_tokens = false;
        self
    }
    pub fn with_include_stop_str_in_output(mut self) -> Self {
        self.include_stop_str_in_output = true;
        self
    }

    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k as i32;
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }
    pub fn with_presence_penalty(mut self, presence_penalty: f32) -> Self {
        self.presence_penalty = presence_penalty;
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

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
