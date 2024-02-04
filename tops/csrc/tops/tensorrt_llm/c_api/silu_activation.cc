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
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <stdexcept>
#include "tops/c_api/c_api.h"
#include "tops/tensorrt_llm/kernels/activation_kernels.h"

template <typename T>
void silu_activation_template(CTensorView a, CTensorView b, int num_token, int inter_size, cudaStream_t stream) {
  T* gating_buf = reinterpret_cast<T*>(a.ptr);
  const T* inter_buf = reinterpret_cast<const T*>(b.ptr);
  fastertransformer::invokeGenericActivation<fastertransformer::SiluActivation, T, T>(gating_buf,
                                                                                      (const T*)nullptr,  // bias
                                                                                      inter_buf,
                                                                                      (const T*)nullptr,  // gated_bias
                                                                                      nullptr,            // ia3_tasks
                                                                                      (const T*)nullptr,  // ia3_weights
                                                                                      num_token,          // m
                                                                                      inter_size,         // n
                                                                                      0,                  // int8_mode
                                                                                      nullptr,  // activation_in
                                                                                      nullptr,  // activation_out
                                                                                      nullptr,  // padding_offset
                                                                                      0,        // seq_len
                                                                                      stream);
}

extern "C" {

void fastertransformer_silu_activation(CTensorView a, CTensorView b, int num_token, int inter_size,
                                       cudaStream_t stream) {
  switch (a.dtype) {
    case ScalarType::DATA_F16: {
      silu_activation_template<half>(a, b, num_token, inter_size, stream);
      break;
    }
    case ScalarType::DATA_BF16: {
      silu_activation_template<__nv_bfloat16>(a, b, num_token, inter_size, stream);
      break;
    }
    case ScalarType::DATA_F32: {
      silu_activation_template<float>(a, b, num_token, inter_size, stream);
      break;
    }
      // case ScalarType::DATA_F64: {
      //   silu_activation_template<double>(a, b, num_token, inter_size, stream);
      //   break;
      // }

    default: {
      throw new std::runtime_error("not supported dtype for silu_activation");
    }
  }
}
}