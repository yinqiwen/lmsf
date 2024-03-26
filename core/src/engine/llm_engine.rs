use anyhow::{anyhow, Result};
use metrics::{counter, gauge};
use std::sync::atomic::AtomicU64;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    thread::JoinHandle,
    time::Instant,
};
use tokio::sync::mpsc::{self, error::TryRecvError};
use tokio::sync::oneshot;

use crate::{
    common::{
        config::{CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig},
        output::RequestOutput,
        sampling_params::{EarlyStopType, SamplingParams},
        sequence::{
            SamplerOutput, Sequence, SequenceGroup, SequenceGroupMetadata, SequenceGroupOutput,
            SequenceGroupRef, SequenceOutput, SequenceRef, SequenceState,
        },
        tokenizer::{decode_token, DecodeTokenOptions},
    },
    model_executor::models::{ChatTemplate, ModelFactory},
    sched::scheduler::{Scheduler, SchedulerOutputs},
    worker::model_worker::Worker,
};

use tokenizers::{Encoding, Tokenizer};

pub enum LLMTaskResponseReceiver {
    Normal(oneshot::Receiver<RequestOutput>),
    Stream(mpsc::Receiver<RequestOutput>),
}
enum LLMTaskResponder {
    Normal(oneshot::Sender<RequestOutput>),
    Stream(mpsc::Sender<RequestOutput>),
}
pub enum LLMPrompt {
    Normal(String),
    MultiRole(Vec<HashMap<String, String>>),
}
impl LLMPrompt {
    pub fn from(s: String) -> Self {
        LLMPrompt::Normal(s)
    }
    pub fn multi_role(s: Vec<HashMap<String, String>>) -> Self {
        LLMPrompt::MultiRole(s)
    }
}
struct LLMTask {
    prompt: LLMPrompt,
    sampling_params: SamplingParams,
    prompt_token_ids: Option<Encoding>,
    arrival_time: std::time::Instant,
    responder: LLMTaskResponder,
}

pub struct LLMEngine {
    cache_config: CacheConfig,
    model_config: Arc<ModelConfig>,
    tokenizer: Tokenizer,
    scheduler: Scheduler,
    worker: Worker,
    chat_template: Option<ChatTemplate>,
    seq_counter: AtomicU64,
}

impl LLMEngine {
    pub fn get_model(&self) -> &str {
        self.model_config.path()
    }

    pub fn from(
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
    ) -> Result<Self> {
        tracing::info!(
            "Initializing an LLM engine with model:{:?}, cache:{:?}, parallel:{:?}, scheduler:{:?}",
            model_config,
            cache_config,
            parallel_config,
            scheduler_config
        );
        let model_config = Arc::new(model_config);

        let model_path = model_config.path();

        let chat_template = ModelFactory::get_chat_template(model_path)?;

        let tokenizer_filename = format!("{}/tokenizer.json", model_path);
        let tokenizer_pretrained = format!("{}/tokenizer.model", model_path);

        // let tokenizer_config_filename = format!("{}/tokenizer_config.json", model_path);
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(|e| {
            tracing::error!("Failed to load tokenizer:{}", e);
            anyhow!("{}", e)
        })?;

        // let tokenizer_config = ModelFactory::load_tokenizer_config(
        //     model_config.inner().get_model_type(),
        //     tokenizer_config_filename,
        // )?;
        // let chat_template = ModelFactory::get_chat_template(
        //     model_config.inner().get_model_type(),
        //     tokenizer_config.get_chat_template(),
        // )?;
        // let tokenizer_ids = TokenizerIds::from(tokenizer_config, &tokenizer)?;
        let mut worker = Worker::from(&cache_config, model_config.clone(), &parallel_config, 0)?;
        tops::reset_random_seed(model_config.seed);
        //worker.profile_run()?;
        worker.init_cache()?;

        tops::reset_random_seed(model_config.seed);

        let scheduler: Scheduler = Scheduler::new(
            &scheduler_config,
            &cache_config,
            worker.get_num_gpu_blocks(),
            worker.get_num_cpu_blocks(),
        );
        Ok(Self {
            cache_config,
            model_config,
            tokenizer,
            // tokenizer_ids,
            scheduler,
            worker,
            chat_template,
            seq_counter: AtomicU64::new(0),
        })
    }

    pub fn add_request(
        &mut self,
        request_id: u64,
        prompt: &str,
        sampling_params: SamplingParams,
        prompt_token_ids: Option<Encoding>,
        arrival_time: std::time::Instant,
    ) -> Result<()> {
        let prompt_token_ids = if let Some(prompt_token_ids) = prompt_token_ids {
            prompt_token_ids
        } else {
            self.tokenizer
                .encode(prompt, true)
                .map_err(|e| anyhow!("{}", e))?
        };

        // # Create the sequences.
        let block_size = self.cache_config.block_size;
        let seq_id = self
            .seq_counter
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let seq = Sequence::new(seq_id, prompt, prompt_token_ids.get_ids(), block_size);
        // tracing::info!(
        //     "##Encode:{:?}, logical_token_blocks:{}",
        //     prompt_token_ids,
        //     seq.logical_token_blocks.len()
        // );
        // # Create the sequence group.
        let seq_group = SequenceGroup::new(request_id, vec![seq], sampling_params, arrival_time);

        // # Add the sequence group to the scheduler.
        self.scheduler.add_seq_group(seq_group);
        Ok(())
    }

    pub fn abort_request(&mut self, request_ids: HashSet<u64>) {
        self.scheduler.abort_seq_group(request_ids);
    }

    pub fn get_num_unfinished_requests(&self) -> usize {
        // """Gets the number of unfinished requests."""
        self.scheduler.get_num_unfinished_seq_groups()
    }

    pub fn has_unfinished_requests(&self) -> bool {
        // """Returns True if there are unfinished requests."""
        self.scheduler.has_unfinished_seqs()
    }

    fn decode_sequence(
        &self,
        seq_mut: &mut Sequence,
        prms: &SamplingParams,
        // arrival_time: &Instant,
    ) -> Result<()> {
        //self.tokenizer.decode(ids, skip_special_tokens);
        // let mut seq_mut = seq.borrow_mut();
        let mut opt = DecodeTokenOptions::default();
        opt.skip_special_tokens = prms.skip_special_tokens;
        opt.read_offset = seq_mut.read_offset;
        opt.prefix_offset = seq_mut.prefix_offset;
        opt.first_decoding = seq_mut.output_text.is_empty();
        // detokenize_incrementally(&self.tokenizer, &seq_mut.get_token_ids(), tokens, opt);
        let (full_txt, new_output_text, prefix_offset, read_offset) =
            decode_token(&self.tokenizer, &seq_mut.get_token_ids(), opt)?;

        seq_mut.prefix_offset = prefix_offset;
        seq_mut.read_offset = read_offset;
        seq_mut.output_text.push_str(new_output_text.as_str());
        // seq_mut.latest_output_token = new_output_text.clone();
        if seq_mut.output_text.len() == 1 {
            // histogram!("prompt.first_token_latnecy_ms")
            //     .record(arrival_time.elapsed().as_millis() as f64);
        } else {
        }

        if !full_txt.is_empty() {
            seq_mut.gen_texts.push(full_txt);
        }
        if !new_output_text.is_empty() {
            seq_mut.gen_texts.push(new_output_text);
        };

        Ok(())
    }
    fn check_stop(&self, seq_mut: &mut Sequence, sampling_params: &SamplingParams) -> bool {
        // let mut seq_mut = seq.borrow_mut();
        for stop_str in &sampling_params.stop {
            if seq_mut.output_text.ends_with(stop_str) {
                if !sampling_params.include_stop_str_in_output {
                    let stop_idx = seq_mut.output_text.len() - stop_str.len();
                    seq_mut.output_text = String::from(&seq_mut.output_text[..stop_idx]);
                    //seq.output_text = String::from(&seq.output_text[:-len(stop_str)])
                }
                seq_mut.update_state(SequenceState::FinishedStopeed);
                return true;
            }
        }
        if let Some(stop_token_ids) = &sampling_params.stop_token_ids {
            if stop_token_ids.contains(&seq_mut.get_last_token_id()) {
                seq_mut.update_state(SequenceState::FinishedStopeed);
                return true;
            }
        }

        if seq_mut.get_len() > self.model_config.get_max_model_len() {
            seq_mut.update_state(SequenceState::FinishedLengthCapped);
            return true;
        }

        if seq_mut.get_output_len() == sampling_params.max_tokens {
            seq_mut.update_state(SequenceState::FinishedLengthCapped);
            return true;
        }

        if !sampling_params.ignore_eos
            && seq_mut.get_last_token_id() == self.model_config.get_eos_token_id()
        {
            seq_mut.update_state(SequenceState::FinishedStopeed);
            return true;
        }
        false
    }
    fn check_beam_search_early_stopping(
        &self,
        sampling_params: &SamplingParams,
        best_running_seq: SequenceRef,
        current_worst_seq: SequenceRef,
    ) -> bool {
        match sampling_params.early_stopping {
            EarlyStopType::EarlyStop(true) => {
                return true;
            }
            _ => {
                //continue
            }
        }
        let length_penalty = sampling_params.length_penalty;
        let current_worst_score = current_worst_seq.borrow().get_beam_search_score(
            length_penalty,
            None,
            Some(self.model_config.get_eos_token_id()),
        );
        let mut highest_attainable_score: f32 = 0.0;
        match sampling_params.early_stopping {
            EarlyStopType::EarlyStop(false) => {
                highest_attainable_score = best_running_seq.borrow().get_beam_search_score(
                    length_penalty,
                    None,
                    Some(self.model_config.get_eos_token_id()),
                );
            }
            EarlyStopType::Never => {
                if length_penalty > 0.0 {
                    let max_possible_length = std::cmp::max(
                        best_running_seq.borrow().get_prompt_len() + sampling_params.max_tokens,
                        self.model_config.get_max_model_len(),
                    );
                    highest_attainable_score = best_running_seq.borrow().get_beam_search_score(
                        length_penalty,
                        Some(max_possible_length),
                        Some(self.model_config.get_eos_token_id()),
                    );
                } else {
                    highest_attainable_score = best_running_seq.borrow().get_beam_search_score(
                        length_penalty,
                        None,
                        Some(self.model_config.get_eos_token_id()),
                    );
                }
            }
            _ => {
                //continue
            }
        }

        current_worst_score >= highest_attainable_score
    }

    fn process_sequence_group_outputs(
        &mut self,
        seq_group: SequenceGroupRef,
        outputs: SequenceGroupOutput,
    ) -> Result<()> {
        if let Some(prompt_logprobs) = outputs.prompt_logprobs {
            seq_group.borrow_mut().set_prompt_logprobs(prompt_logprobs);
        }

        let parent_seqs = seq_group.borrow().get_seqs(Some(SequenceState::Running));
        let existing_finished_seqs = seq_group.borrow().get_finished_seqs();
        let mut parent_child_dict: HashMap<u64, Vec<SequenceOutput>> = HashMap::new();

        for seq in &parent_seqs {
            parent_child_dict.insert(seq.borrow().seq_id, Vec::new());
        }
        for sample in outputs.samples {
            if let Some(entry) = parent_child_dict.get_mut(&sample.parent_seq_id) {
                entry.push(sample);
            }
        }

        let mut child_seqs: Vec<(SequenceRef, SequenceRef)> = Vec::new();
        for parent in parent_seqs {
            let mut parent_borrow = parent.borrow_mut();
            if let Some(child_samples) = parent_child_dict.get(&parent_borrow.seq_id) {
                if child_samples.is_empty() {
                    parent_borrow.update_state(SequenceState::FinishedAborted);
                    let _ = seq_group.borrow_mut().remove(parent_borrow.seq_id);
                    self.scheduler.free_seq(parent_borrow.seq_id);
                    continue;
                }

                for child_sample in child_samples.iter().take(child_samples.len() - 1) {
                    //println!("{}", element);
                    let new_child_seq_id = self
                        .seq_counter
                        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let mut child: Sequence = parent_borrow.fork(new_child_seq_id);
                    child.append_token_id(child_sample.output_token, child_sample.logprobs.clone());
                    //child_seqs.push((child.newRef(), parent.clone()));
                    child_seqs.push((child.new_ref(), parent.clone()))
                }
                let last_child_sample = child_samples.last().unwrap();
                parent_borrow.append_token_id(
                    last_child_sample.output_token,
                    last_child_sample.logprobs.clone(),
                );
                //child_seqs.push((parent.clone(), parent.clone()));
                child_seqs.push((parent.clone(), parent.clone()))
            }
        }

        let mut seq_group_borrow = seq_group.borrow_mut();

        for (seq, _) in &mut child_seqs {
            let mut_seq = &mut seq.borrow_mut();
            self.decode_sequence(
                mut_seq,
                &seq_group_borrow.sampling_params,
                // &seq_group_borrow.arrival_time,
            )?;
            if self.check_stop(mut_seq, &seq_group_borrow.sampling_params) {
                tracing::info!("seq stop:{:?}", mut_seq.get_state());
            }
        }
        if !seq_group_borrow.sampling_params.use_beam_search {
            for (seq, parent) in &child_seqs {
                if !std::rc::Rc::ptr_eq(seq, parent) {
                    let _ = seq_group_borrow.add_ref(seq.clone());
                    if !seq.borrow().is_finished() {
                        self.scheduler.fork_seq(&parent.borrow(), &seq.borrow());
                    }
                }
            }
            for (seq, parent) in &child_seqs {
                if std::rc::Rc::ptr_eq(seq, parent) {
                    if seq.borrow().is_finished() {
                        self.scheduler.free_seq(seq.borrow().seq_id);
                    }
                }
            }
            return Ok(());
        }

        let mut selected_child_seqs = Vec::new();
        let mut unselected_child_seqs = Vec::new();
        let beam_width = seq_group_borrow.sampling_params.best_of.unwrap();
        let eos_token_id = self.model_config.get_eos_token_id();
        let length_penalty = seq_group_borrow.sampling_params.length_penalty;
        let mut existing_finished_seqs = existing_finished_seqs
            .into_iter()
            .map(|x| (x.clone(), None, false))
            .collect::<Vec<(SequenceRef, Option<SequenceRef>, bool)>>();
        let mut new_finished_seqs = child_seqs
            .iter()
            .map(|(seq, parent)| (seq.clone(), Some(parent.clone()), true))
            .collect::<Vec<(SequenceRef, Option<SequenceRef>, bool)>>();
        existing_finished_seqs.append(&mut new_finished_seqs);
        let mut all_finished_seqs = existing_finished_seqs;
        all_finished_seqs.sort_by(|x, y| {
            let x_score =
                x.0.borrow()
                    .get_beam_search_score(length_penalty, None, Some(eos_token_id));
            let y_score =
                y.0.borrow()
                    .get_beam_search_score(length_penalty, None, Some(eos_token_id));
            x_score.total_cmp(&y_score).reverse()
        });

        for (seq, parent, is_new) in &all_finished_seqs[..beam_width] {
            if *is_new {
                selected_child_seqs.push((seq.clone(), parent.clone()));
            }
        }
        for (seq, parent, is_new) in &all_finished_seqs[beam_width..] {
            if *is_new {
                unselected_child_seqs.push((seq.clone(), parent.clone()));
            } else {
                let _ = seq_group_borrow.remove(seq.borrow().seq_id);
            }
        }

        let mut running_child_seqs = child_seqs
            .iter()
            .filter(|(seq, _)| !seq.borrow().is_finished())
            .map(|(seq, parent)| (seq.clone(), Some(parent.clone())))
            .collect::<Vec<_>>();
        running_child_seqs.sort_by(|x, y| {
            let x_score =
                x.0.borrow()
                    .get_beam_search_score(length_penalty, None, Some(eos_token_id));
            let y_score =
                y.0.borrow()
                    .get_beam_search_score(length_penalty, None, Some(eos_token_id));
            x_score.total_cmp(&y_score).reverse()
        });
        let mut stop_beam_search = false;
        if running_child_seqs.is_empty() {
            stop_beam_search = true;
        } else if all_finished_seqs.len() < beam_width {
            stop_beam_search = false;
        } else {
            let best_running_seq = running_child_seqs[0].0.clone();
            let current_worst_seq = all_finished_seqs[beam_width - 1].0.clone();
            stop_beam_search = self.check_beam_search_early_stopping(
                &seq_group_borrow.sampling_params,
                best_running_seq,
                current_worst_seq,
            );
        }

        if stop_beam_search {
            unselected_child_seqs.extend(running_child_seqs);
        } else {
            selected_child_seqs.extend_from_slice(&running_child_seqs[..beam_width]);
            unselected_child_seqs.extend_from_slice(&running_child_seqs[beam_width..])
        }

        for (seq, parent) in &selected_child_seqs {
            if let Some(parent) = parent {
                if !std::rc::Rc::ptr_eq(seq, parent) {
                    let _ = seq_group_borrow.add_ref(seq.clone());
                    if !seq.borrow().is_finished() {
                        self.scheduler.fork_seq(&parent.borrow(), &seq.borrow());
                    }
                }
            }
        }

        for (seq, parent) in selected_child_seqs {
            if let Some(parent) = parent {
                if std::rc::Rc::ptr_eq(&seq, &parent) && seq.borrow().is_finished() {
                    self.scheduler.free_seq(seq.borrow().seq_id);
                }
            }
        }

        for (seq, parent) in unselected_child_seqs {
            if let Some(parent) = parent {
                if std::rc::Rc::ptr_eq(&seq, &parent) {
                    let seq_id = seq.borrow().seq_id;
                    let _ = seq_group_borrow.remove(seq_id);
                    self.scheduler.free_seq(seq_id);
                }
            }
        }

        Ok(())
    }

    fn process_model_outputs(
        &mut self,
        output: SamplerOutput,
        scheduler_outputs: SchedulerOutputs,
    ) -> Result<Vec<RequestOutput>> {
        let now = Instant::now();
        for (seq_group, outputs) in scheduler_outputs
            .scheduled_seq_groups
            .iter()
            .zip(output.into_iter())
        {
            self.process_sequence_group_outputs(seq_group.clone(), outputs)?;
        }

        self.scheduler.free_finished_seq_groups();

        let mut request_outputs = Vec::new();
        for seq_group in scheduler_outputs.scheduled_seq_groups {
            seq_group.borrow_mut().maybe_set_first_token_time(now);
            let request_output = RequestOutput::from(&seq_group);
            request_outputs.push(request_output);
        }
        for seq_group in scheduler_outputs.ignored_seq_groups {
            let request_output = RequestOutput::from(&seq_group);
            request_outputs.push(request_output);
        }

        Ok(request_outputs)
    }

    pub fn step(&mut self) -> Result<Vec<RequestOutput>> {
        let sched_result: SchedulerOutputs = self.scheduler.schedule()?;
        if sched_result.is_empty() {
            let mut outputs = Vec::new();
            for seq_group in sched_result.ignored_seq_groups {
                outputs.push(RequestOutput::from(&seq_group));
            }
            return Ok(outputs);
        }

        let now = Instant::now();
        let mut seq_group_metadata_list: Vec<SequenceGroupMetadata> = Vec::new();
        for seq_group in &sched_result.scheduled_seq_groups {
            seq_group.borrow_mut().maybe_set_first_scheduled_time(now);

            let mut seq_data = HashMap::new();
            let mut block_tables = HashMap::new();
            let seq_group = seq_group.borrow();
            for seq in seq_group.get_seqs(Some(SequenceState::Running)) {
                let seq_borrow = seq.borrow();
                let seq_id = seq_borrow.seq_id;
                //seq_data.insert(seq_id, seq.clone());
                seq_data.insert(seq_id, seq_borrow.data.clone());
                let blocks = self.scheduler.get_block_table(seq_borrow.seq_id);
                block_tables.insert(seq_id, blocks);
            }
            let seq_group_metadata = SequenceGroupMetadata::new(
                seq_group.request_id,
                sched_result.prompt_run,
                seq_data,
                seq_group.sampling_params.clone(),
                block_tables,
            );
            seq_group_metadata_list.push(seq_group_metadata);
        }
        let _modle_start = std::time::Instant::now();
        let output = self
            .worker
            .execute_model(seq_group_metadata_list, &sched_result, true)?;

        // tracing::info!("model step exec cost {:?}", modle_start.elapsed());
        if let Some(output) = output {
            self.process_model_outputs(output, sched_result)
        } else {
            Ok(Vec::new())
        }
    }

    fn get_prompt(&self, prompt: &LLMPrompt) -> anyhow::Result<String> {
        match prompt {
            LLMPrompt::Normal(s) => Ok(String::from(s.as_str())),
            LLMPrompt::MultiRole(s) => match &self.chat_template {
                Some(template) => template.apply(s),
                None => Err(anyhow!("empty chat template")),
            },
        }
    }

    pub fn add_prompt(
        &mut self,
        request_id: u64,
        prompt: &LLMPrompt,
        sampling_params: SamplingParams,
        prompt_token_ids: Option<Encoding>,
        arrival_time: std::time::Instant,
    ) -> Result<()> {
        let prompt = self.get_prompt(prompt)?;
        // println!("##prompt:{}", prompt);
        self.add_request(
            request_id,
            &prompt,
            sampling_params,
            prompt_token_ids,
            arrival_time,
        )
    }
}

fn run_engine(
    mut engine: LLMEngine,
    notify_sender: oneshot::Sender<()>,
    mut receiver: mpsc::UnboundedReceiver<Option<LLMTask>>,
) -> Result<()> {
    let mut num_unfinished_requests = 0_usize;
    let max_num_seqs = engine.scheduler.scheduler_config.max_num_seqs;
    let mut task_map = HashMap::new();
    let mut request_id = 1_u64;

    let _ = notify_sender.send(());

    loop {
        while num_unfinished_requests < max_num_seqs {
            let task = if num_unfinished_requests == 0 {
                match receiver.blocking_recv() {
                    Some(Some(task)) => task,
                    Some(None) => {
                        return Ok(());
                    }
                    None => {
                        return Ok(());
                    }
                }
            } else {
                match receiver.try_recv() {
                    Ok(Some(task)) => task,
                    Ok(None) => {
                        return Ok(());
                    }
                    Err(TryRecvError::Empty) => {
                        break;
                    }
                    Err(TryRecvError::Disconnected) => {
                        return Err(anyhow!("receiver disconnected"));
                    }
                }
            };

            match engine.add_prompt(
                request_id,
                &task.prompt,
                task.sampling_params.clone(),
                task.prompt_token_ids.clone(),
                task.arrival_time,
            ) {
                Ok(()) => {
                    task_map.insert(request_id, task);
                }
                Err(e) => {
                    tracing::error!("Failed to add prompt:{} with error:{}", request_id, e);
                }
            }
            request_id += 1;
            num_unfinished_requests = engine.get_num_unfinished_requests();
        }

        if engine.has_unfinished_requests() {
            let outputs = engine.step()?;

            for req_out in outputs {
                let request_id = req_out.request_id;
                if req_out.finished {
                    if let Some(task) = task_map.remove(&request_id) {
                        match task.responder {
                            LLMTaskResponder::Stream(r) => {
                                let _ = r.blocking_send(req_out);
                            }
                            LLMTaskResponder::Normal(r) => {
                                let _ = r.send(req_out);
                            }
                        }
                    } else {
                        tracing::error!("No request_id:{} found in task map", request_id);
                    }
                } else {
                    if let Some(task) = task_map.get(&request_id) {
                        match &task.responder {
                            LLMTaskResponder::Stream(r) => {
                                let _result = r.blocking_send(req_out);
                                // tracing::info!("#### send:{:?}", result);
                            }
                            _ => {}
                        }
                    } else {
                        tracing::error!("No request_id:{} found in task map", request_id);
                    }
                }
            }
            num_unfinished_requests = engine.get_num_unfinished_requests();
        }
    }
}

struct JoinHandleWrapper {
    h: Option<JoinHandle<()>>,
}

impl JoinHandleWrapper {
    fn new(h: JoinHandle<()>) -> Arc<Self> {
        Arc::new(Self { h: Some(h) })
    }
}

impl Drop for JoinHandleWrapper {
    fn drop(&mut self) {
        match self.h.take() {
            Some(h) => {
                let _ = h.join();
            }
            _ => {}
        }
    }
}

#[derive(Clone)]
pub struct AsyncLLMEngine {
    model: String,
    sender: mpsc::UnboundedSender<Option<LLMTask>>,
    runner: Arc<JoinHandleWrapper>,
}

impl Drop for AsyncLLMEngine {
    fn drop(&mut self) {
        let _ = self.sender.send(None);
    }
}

impl AsyncLLMEngine {
    pub async fn new(
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        profiler_enable: bool,
    ) -> Self {
        let (notify_sender, notify_rx) = oneshot::channel::<()>();
        let (s, receiver) = mpsc::unbounded_channel();
        let model = String::from(model_config.path());
        let h: JoinHandle<()> = std::thread::spawn(move || {
            match LLMEngine::from(
                model_config,
                cache_config,
                parallel_config,
                scheduler_config,
            ) {
                Ok(engine) => {
                    if profiler_enable {
                        common::cuda_ext::cuda_profiler_start();
                    }
                    if let Err(e) = run_engine(engine, notify_sender, receiver) {
                        tracing::error!("run_engine error:{}", e);
                    }
                    if profiler_enable {
                        common::cuda_ext::cuda_profiler_stop();
                    }
                }
                Err(e) => {
                    tracing::error!("create engine failed:{}", e);
                }
            }
        });
        let _ = notify_rx.await;
        Self {
            model,
            sender: s,
            runner: JoinHandleWrapper::new(h),
        }
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn add(
        &self,
        prompt: LLMPrompt,
        sampling_params: SamplingParams,
        stream: bool,
    ) -> Result<LLMTaskResponseReceiver> {
        counter!("llm.prompt_recv").increment(1);
        gauge!("llm.alive_prompt").increment(1.0);

        let (recv, responder) = if stream {
            let (s, r) = mpsc::channel::<RequestOutput>(16);
            (
                LLMTaskResponseReceiver::Stream(r),
                LLMTaskResponder::Stream(s),
            )
        } else {
            let (s, r) = oneshot::channel::<RequestOutput>();
            (
                LLMTaskResponseReceiver::Normal(r),
                LLMTaskResponder::Normal(s),
            )
        };
        let arrival_time = std::time::Instant::now();
        let task = LLMTask {
            prompt,
            sampling_params,
            prompt_token_ids: None,
            arrival_time,
            responder,
        };
        self.sender
            .send(Some(task))
            .map_err(|e| anyhow!("add llm task:{}", e))?;
        Ok(recv)
    }
}
