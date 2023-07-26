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
#include "tinfer/log/log.h"
#include "tinfer/module/module.h"

MODULE_BEGIN(prepare_inputs)

TENSOR_OUTPUT(input_ids)
TENSOR_OUTPUT(input_lengths)
TENSOR_OUTPUT(output_seq_len)

TENSOR_OUTPUT(temperature)
TENSOR_OUTPUT(len_penalty)
TENSOR_OUTPUT(min_length)
TENSOR_OUTPUT(start_id)
TENSOR_OUTPUT(end_id)
TENSOR_OUTPUT(repetition_penalty)
TENSOR_OUTPUT(presence_penalty)
TENSOR_OUTPUT(runtime_top_k)
TENSOR_OUTPUT(runtime_top_p)
TENSOR_OUTPUT(beam_search_diversity_rate)

TENSOR_OUTPUT(output_ids)
TENSOR_OUTPUT(sequence_length)
TENSOR_OUTPUT(response_input_lengths)
TENSOR_OUTPUT(is_finished)

void Setup(Context& ctx) override {}
void Forward(Context& ctx) override {
  const size_t request_batch_size = ctx.task->start_lengths.size();
  size_t beam_width = ctx.task->beam_width;

  if (beam_width != 1 && beam_width != 2 && beam_width != 3 && beam_width != 4 && beam_width != 8 && beam_width != 16 &&
      beam_width != 32) {
    TINFER_WARN("beam_width = {} is invalid. Set it to 1 to use sampling by default.", beam_width);
    beam_width = 1;
  }
  size_t total_length = ctx.task->max_request_output_len + ctx.task->max_input_length;

  input_ids.Set<int>(ctx.allocator.get(), std::vector<size_t>{request_batch_size, (size_t)ctx.task->max_input_length},
                     ctx.task->start_ids.data());
  input_lengths.Set<int>(ctx.allocator.get(), std::vector<size_t>{request_batch_size}, ctx.task->start_lengths.data());
  temperature.Set<float>(std::vector<size_t>{request_batch_size}, ctx.task->temperature.data());
  len_penalty.Set<float>(std::vector<size_t>{request_batch_size}, ctx.task->len_penalty.data());
  if (!ctx.task->min_length.empty()) {
    min_length.Set<uint32_t>(std::vector<size_t>{ctx.task->min_length.size()}, ctx.task->min_length.data());
  }
  if (!ctx.task->start_id.empty()) {
    start_id.Set<uint32_t>(std::vector<size_t>{ctx.task->start_id.size()}, ctx.task->start_id.data());
  }
  if (!ctx.task->end_id.empty()) {
    end_id.Set<uint32_t>(std::vector<size_t>{ctx.task->end_id.size()}, ctx.task->end_id.data());
  }

  output_seq_len.Set<int>(ctx.allocator.get(), std::vector<size_t>{request_batch_size});

  if (!ctx.task->repetition_penalty.empty()) {
    repetition_penalty.Set<uint32_t>(std::vector<size_t>{ctx.task->repetition_penalty.size()},
                                     ctx.task->repetition_penalty.data());
  }
  if (!ctx.task->presence_penalty.empty()) {
    presence_penalty.Set<uint32_t>(std::vector<size_t>{ctx.task->presence_penalty.size()},
                                   ctx.task->presence_penalty.data());
  }
  if (!ctx.task->runtime_top_k.empty()) {
    runtime_top_k.Set<uint32_t>(std::vector<size_t>{ctx.task->runtime_top_k.size()}, ctx.task->runtime_top_k.data());
  }
  if (!ctx.task->runtime_top_p.empty()) {
    runtime_top_k.Set<float>(std::vector<size_t>{ctx.task->runtime_top_p.size()}, ctx.task->runtime_top_p.data());
  }
  if (!ctx.task->beam_search_diversity_rate.empty()) {
    beam_search_diversity_rate.Set<float>(std::vector<size_t>{ctx.task->beam_search_diversity_rate.size()},
                                          ctx.task->beam_search_diversity_rate.data());
  }

  output_ids.Set<uint32_t>(ctx.allocator.get(), std::vector<size_t>{request_batch_size, beam_width, total_length});
  sequence_length.Set<int32_t>(ctx.allocator.get(), std::vector<size_t>{request_batch_size, beam_width});
  response_input_lengths.Set<int32_t>(ctx.allocator.get(), std::vector<size_t>{request_batch_size, beam_width});
  is_finished.Set<bool>(ctx.allocator.get(), std::vector<size_t>{request_batch_size, beam_width});
}
MODULE_END