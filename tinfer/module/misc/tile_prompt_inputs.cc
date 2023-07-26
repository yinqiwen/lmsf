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

MODULE_BEGIN(tile_prompt_inputs)
TENSOR_INPUT(input_ids)
TENSOR_INPUT(input_lengths)
TENSOR_INPUT(request_prompt_lengths)

TENSOR_OUTPUT(tiled_input_ids)
TENSOR_OUTPUT(tiled_input_lengths)
TENSOR_OUTPUT(tiled_prompt_lengths)

// int* tiled_input_ids_buf_ = nullptr;
// int* tiled_input_lengths_buf_ = nullptr;
// int* tiled_prompt_lengths_buf_ = nullptr;  // only needed by prefix prompts
void Setup(Context& ctx) override {}
void Forward(Context& ctx) override {
  tiled_input_ids.Set<int>(ctx.allocator.get(),
                           std::vector<size_t>{ctx.task->batch_size, ctx.task->beam_width, ctx.task->max_session_len});
  tiled_input_lengths.Set<int>(ctx.allocator.get(), std::vector<size_t>{ctx.task->batch_size, ctx.task->beam_width});
  tiled_prompt_lengths.Set<int>(ctx.allocator.get(), std::vector<size_t>{ctx.task->batch_size, ctx.task->beam_width});
  tinfer::invokeTileGptPromptInputs(
      tiled_input_ids.GetPtr<int>(), tiled_input_lengths.GetPtr<int>(), tiled_prompt_lengths.GetPtr<int>(),
      input_ids->GetPtr<int>(), input_lengths->GetPtr<const int>(),
      nullptr != request_prompt_lengths ? request_prompt_lengths->GetPtr<const int>() : nullptr, ctx.task->batch_size,
      ctx.task->beam_width, ctx.task->max_input_length, ctx.stream);
}
MODULE_END