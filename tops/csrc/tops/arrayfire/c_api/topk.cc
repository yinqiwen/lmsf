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

#include "tops/arrayfire/ops/topk.h"
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include "tops/arrayfire/common/array.h"
#include "tops/c_api/c_api.h"
extern "C" {
void cuda_topk_tensor(CTensorView input, int k, int dim, int topk_type, cudaStream_t stream, CTensorView output,
                      CTensorView indices) {
  auto indices_array = arrayfire::cuda::Array<uint32_t>::create(indices);
  switch (input.dtype) {
    case ScalarType::DATA_F16: {
      auto input_array = arrayfire::cuda::Array<__half>::create(input);
      auto output_array = arrayfire::cuda::Array<__half>::create(output);
      arrayfire::cuda::topk(output_array, indices_array, input_array, k, dim,
                            static_cast<arrayfire::topkFunction>(topk_type), stream);
      break;
    }
    case ScalarType::DATA_F32: {
      auto input_array = arrayfire::cuda::Array<float>::create(input);
      auto output_array = arrayfire::cuda::Array<float>::create(output);
      arrayfire::cuda::topk(output_array, indices_array, input_array, k, dim,
                            static_cast<arrayfire::topkFunction>(topk_type), stream);
      break;
    }
    case ScalarType::DATA_F64: {
      auto input_array = arrayfire::cuda::Array<double>::create(input);
      auto output_array = arrayfire::cuda::Array<double>::create(output);
      arrayfire::cuda::topk(output_array, indices_array, input_array, k, dim,
                            static_cast<arrayfire::topkFunction>(topk_type), stream);
      break;
    }
    default: {
      throw new std::runtime_error("not supported dtype for cuda_sort_tensor");
    }
  }
}
}