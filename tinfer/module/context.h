/*
** BSD 3-Clause License
**
** Copyright (c) 2023, qiyingwang <qiyingwang@tencent.com>, the respective contributors, as shown by the AUTHORS file.
** All rights reserved.
**
** Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are met:
** * Redistributions of source code must retain the above copyright notice, this
** list of conditions and the following disclaimer.
**
** * Redistributions in binary form must reproduce the above copyright notice,
** this list of conditions and the following disclaimer in the documentation
** and/or other materials provided with the distribution.
**
** * Neither the name of the copyright holder nor the names of its
** contributors may be used to endorse or promote products derived from
** this software without specific prior written permission.
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
** AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
** IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
** DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
** FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
** DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
** SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
** CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
** OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "tinfer/model/model_task.h"
#include "tinfer/model/model_weights.h"
#include "tinfer/module/params.h"
#include "tinfer/tensor/tensor.h"
#include "tinfer/utils/cuda_allocator.h"

namespace tinfer {

struct BatchGenerateTask {
  std::vector<GenerateTask> batch;

  uint32_t batch_size = 1;
  uint32_t beam_width = 1;
  bool stream = false;

  std::vector<int> start_lengths;
  std::vector<int> start_ids;
  int max_input_length = 0;
  size_t max_request_output_len = 0;
  size_t max_session_len = 0;

  // input_tensors:
  //      input_ids [batch_size, max_input_length]
  //      input_lengths [batch_size]
  //      input_lengths_h [batch_size] on cpu, optional
  //      prompt_learning_task_name_ids [batch_size] on cpu
  //      output_seq_len [batch_size] on cpu
  //      stop_words_list [batch_size, 2, stop_words_length], optional
  //      bad_words_list [2, bad_words_length] or [batch_size, 2, bad_words_length], optional
  //      start_id [batch_size] on cpu, optional
  //      end_id [batch_size] on cpu, optional
  //      runtime_top_k [1] or [batch_size] on cpu, optional, uint.
  //      runtime_top_p [1] or [batch_size] on cpu, optional, float.
  //      beam_search_diversity_rate [1] or [batch_size] on cpu, optional, float.
  //      temperature [1] or [batch_size] on cpu, optional, float.
  //      len_penalty [1] or [batch_size] on cpu, optional, float.
  //      repetition_penalty [1] or [batch_size] on cpu, optional, float.
  //      presence_penalty [1] or [batch_size] on cpu, optional, float.
  //          Only one of repetition and presence penalties is allowed.
  //      min_length [1] or [batch_size] on cpu, optional, int
  //      random_seed [1] or [batch_size] on cpu, optional, unsigned long long int.
  //      request_prompt_lengths [batch_size], optional
  //      request_prompt_lengths_h [batch_size], cpu, optional
  //      request_prompt_embedding [batch_size, max_prompt_length, hidden_units], float, optional
  //      request_prompt_type [batch_size], int, optional
  //      is_return_context_cum_log_probs [1] on cpu, bool, optional
  //      session_len [1] on cpu, uint32, optional
  //      memory_len [1] on cpu, uint32, optional
  //      continue_gen [1] on cpu, bool, optional
  //      is_return_context_embeddings [1] on cpu, bool, optional
  //      top_p_decay [batch_size] on gpu, float, optional
  //      top_p_min [batch_size] on gpu, float, optional
  //      top_p_reset_ids [batch_size] on gpu, uint32, optional
  //      repetition_penalty_ignore_orig_input [1] or [batch_size] on cpu, optional

  std::vector<int> output_seq_len;
  std::vector<int> start_id;
  std::vector<int> end_id;
  std::vector<int> runtime_top_k;
  std::vector<float> runtime_top_p;
  std::vector<float> beam_search_diversity_rate;
  std::vector<float> temperature;
  std::vector<float> len_penalty;
  std::vector<float> repetition_penalty;
  std::vector<float> presence_penalty;
  std::vector<int> min_length;
  std::vector<uint64_t> random_seed;

  std::vector<int> stream_gen_lengths;
};

struct Context {
  std::unique_ptr<BatchGenerateTask> task;
  TensorMap tensors;

  cudaStream_t stream = nullptr;
  cublasHandle_t cublas_handle;
  cublasLtHandle_t cublaslt_handle;
#ifdef SPARSITY_ENABLED
  cusparseLtHandle_t cusparselt_handle;
#else
#endif
  std::unique_ptr<CudaAllocator> allocator;

  ModelConfig model_config;
  const ModelWeights* weights = nullptr;

  int Init(const ModelWeights* model_weights);

  Tensor* GetTensor(const std::string& name);
  void InsertTenseor(const std::string& key, const Tensor& value);
  const void* GetWeight(const std::string& name) const;
};
}  // namespace tinfer