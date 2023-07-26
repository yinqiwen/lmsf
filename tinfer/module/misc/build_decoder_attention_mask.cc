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

GENERIC_MODULE_BEGIN(build_decoder_attention_mask)
TENSOR_INPUT(tiled_input_lengths)
TENSOR_OUTPUT(input_attention_mask)

void Setup(Context& ctx) override {}
void Forward(Context& ctx) override {
  input_attention_mask.Set<T>(
      ctx.allocator.get(), std::vector<size_t>{ctx.task->batch_size * ctx.task->beam_width,
                                               (size_t)ctx.task->max_input_length, (size_t)ctx.task->max_input_length});
  tinfer::invokeBuildDecoderAttentionMask<T>(input_attention_mask.GetPtr<T>(), tiled_input_lengths->GetPtr<int>(),
                                             nullptr, ctx.task->batch_size * ctx.task->beam_width,
                                             ctx.task->max_input_length, 0, ctx.stream);
}
GENERIC_MODULE_END