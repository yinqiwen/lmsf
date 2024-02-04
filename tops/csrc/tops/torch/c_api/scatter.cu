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
#include <vector>
#include "tops/arrayfire/common/array.h"
#include "tops/arrayfire/ops/tile.h"
#include "tops/c_api/c_api.h"
#include "tops/torch/kernel/scatter_gather.cuh"

extern "C" {

void cuda_scatter_tensor(CTensorView index, CTensorView src, int64_t dim, CTensorView dst, cudaStream_t stream) {
  switch (src.dtype) {
    case ScalarType::DATA_F16: {
      switch (index.dtype) {
        case ScalarType::DATA_U8: {
          at::native::cuda_scatter_gather_base_kernel<half, uint8_t, true, true>()(dst, dim, index, src,
                                                                                   at::native::tensor_assign, stream);
          break;
        }
        case ScalarType::DATA_U32: {
          at::native::cuda_scatter_gather_base_kernel<half, uint32_t, true, true>()(dst, dim, index, src,
                                                                                    at::native::tensor_assign, stream);
          break;
        }
        case ScalarType::DATA_I64: {
          at::native::cuda_scatter_gather_base_kernel<half, int64_t, true, true>()(dst, dim, index, src,
                                                                                   at::native::tensor_assign, stream);
          break;
        }
        default: {
          throw new std::runtime_error("not supported index dtype for cuda_scatter(u8/u32/i64)");
        }
      }
      break;
    }
    case ScalarType::DATA_F32: {
      switch (index.dtype) {
        case ScalarType::DATA_U8: {
          at::native::cuda_scatter_gather_base_kernel<float, uint8_t, true, true>()(dst, dim, index, src,
                                                                                    at::native::tensor_assign, stream);
          break;
        }
        case ScalarType::DATA_U32: {
          at::native::cuda_scatter_gather_base_kernel<float, uint32_t, true, true>()(dst, dim, index, src,
                                                                                     at::native::tensor_assign, stream);
          break;
        }
        case ScalarType::DATA_I64: {
          at::native::cuda_scatter_gather_base_kernel<float, int64_t, true, true>()(dst, dim, index, src,
                                                                                    at::native::tensor_assign, stream);
          break;
        }
        default: {
          throw new std::runtime_error("not supported index dtype for cuda_scatter(u8/u32/i64)");
        }
      }

      break;
    }
    case ScalarType::DATA_U8: {
      switch (index.dtype) {
        case ScalarType::DATA_U8: {
          at::native::cuda_scatter_gather_base_kernel<uint8_t, uint8_t, true, true>()(
              dst, dim, index, src, at::native::tensor_assign, stream);
          break;
        }
        case ScalarType::DATA_U32: {
          at::native::cuda_scatter_gather_base_kernel<uint8_t, uint32_t, true, true>()(
              dst, dim, index, src, at::native::tensor_assign, stream);
          break;
        }
        case ScalarType::DATA_I64: {
          at::native::cuda_scatter_gather_base_kernel<uint8_t, int64_t, true, true>()(
              dst, dim, index, src, at::native::tensor_assign, stream);
          break;
        }
        default: {
          throw new std::runtime_error("not supported index dtype for cuda_scatter(u8/u32/i64)");
        }
      }

      break;
    }
    case ScalarType::DATA_U32: {
      switch (index.dtype) {
        case ScalarType::DATA_U8: {
          at::native::cuda_scatter_gather_base_kernel<uint32_t, uint8_t, true, true>()(
              dst, dim, index, src, at::native::tensor_assign, stream);
          break;
        }
        case ScalarType::DATA_U32: {
          at::native::cuda_scatter_gather_base_kernel<uint32_t, uint32_t, true, true>()(
              dst, dim, index, src, at::native::tensor_assign, stream);
          break;
        }
        case ScalarType::DATA_I64: {
          at::native::cuda_scatter_gather_base_kernel<uint32_t, int64_t, true, true>()(
              dst, dim, index, src, at::native::tensor_assign, stream);
          break;
        }
        default: {
          throw new std::runtime_error("not supported index dtype for cuda_scatter(u8/u32/i64)");
        }
      }

      break;
    }
    case ScalarType::DATA_I64: {
      switch (index.dtype) {
        case ScalarType::DATA_U8: {
          at::native::cuda_scatter_gather_base_kernel<int64_t, uint8_t, true, true>()(
              dst, dim, index, src, at::native::tensor_assign, stream);
          break;
        }
        case ScalarType::DATA_U32: {
          at::native::cuda_scatter_gather_base_kernel<int64_t, uint32_t, true, true>()(
              dst, dim, index, src, at::native::tensor_assign, stream);
          break;
        }
        case ScalarType::DATA_I64: {
          at::native::cuda_scatter_gather_base_kernel<int64_t, int64_t, true, true>()(
              dst, dim, index, src, at::native::tensor_assign, stream);
          break;
        }
        default: {
          throw new std::runtime_error("not supported index dtype for cuda_scatter(u8/u32/i64)");
        }
      }

      break;
    }
    default: {
      throw new std::runtime_error("not supported src dtype for cuda_create_exponential_tensor");
    }
  }
}
}
