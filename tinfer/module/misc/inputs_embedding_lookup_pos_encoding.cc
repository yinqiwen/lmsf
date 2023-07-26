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
#include "tinfer/module/module.h"

#include "tinfer/kernel/gpt_kernels.h"

GENERIC_MODULE_BEGIN(inputs_embedding_lookup_pos_encoding)
TENSOR_INPUT(output_ids)

TENSOR_INPUT(input_lengths)
TENSOR_INPUT(request_prompt_lengths)

TENSOR_INPUT(tiled_input_ids)
TENSOR_INPUT(request_prompt_embedding)

TENSOR_OUTPUT(context_decoder_input)
TENSOR_OUTPUT(context_decoder_output)

void Setup(Context& ctx) override {}
void Forward(Context& ctx) override {
  context_decoder_input.Set<T>(ctx.allocator.get(),
                               std::vector<size_t>{ctx.task->batch_size, ctx.task->beam_width,
                                                   ctx.task->max_input_length * ctx.model_config.hidden_units});

  context_decoder_output.Set<T>(ctx.allocator.get(),
                                std::vector<size_t>{ctx.task->batch_size, ctx.task->beam_width,
                                                    ctx.task->max_input_length * ctx.model_config.hidden_units});

  int prompt_learning_start_id = 0;
  int max_request_p_prompt_length = 0;
  bool use_request_p_prompt_embedding = false;
  tinfer::pPromptTuningParam<T> prompt_param{
      (const T**)nullptr, prompt_learning_start_id, max_request_p_prompt_length, use_request_p_prompt_embedding,
      nullptr != request_prompt_embedding ? request_prompt_embedding->GetPtr<T>() : nullptr};
  auto* position_encoding_table = ctx.GetWeight("wpe");
  auto* pre_decoder_embedding_table = ctx.GetWeight("wte");
  tinfer::invokeInputIdsEmbeddingLookupPosEncoding<T>(
      context_decoder_input.GetPtr<T>(), output_ids->GetPtr<int>(), (T*)pre_decoder_embedding_table,
      (T*)position_encoding_table, prompt_param, tiled_input_ids->GetPtr<int>(), 1, ctx.task->max_input_length,
      ctx.task->max_input_length, ctx.task->batch_size * ctx.task->beam_width, ctx.model_config.hidden_units,
      ctx.stream);
}
GENERIC_MODULE_END