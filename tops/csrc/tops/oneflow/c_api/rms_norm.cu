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
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "tops/c_api/c_api.h"
#include "tops/common/err_cuda.h"
#include "tops/oneflow/kernel/rms_norm.cuh"

namespace oneflow {
namespace cuda {
namespace rms_norm {

template <typename SRC, typename DST, bool affine>
struct AffineStore {
  AffineStore(DST* dst, const DST* weight, int32_t row_size) : dst(dst), weight(weight), row_size(row_size) {}

  template <int N>
  __device__ void store(const SRC* src, int32_t row, int32_t col) {
    layer_norm::Pack<DST, N> dst_pack;
    layer_norm::Pack<DST, N> weight_pack;
    const int32_t offset = (row * row_size + col) / N;
    const int32_t weight_offset = col / N;
    if (affine) {
      weight_pack.storage = *(reinterpret_cast<const layer_norm::PackType<DST, N>*>(weight) + weight_offset);
    }
#pragma unroll
    for (int i = 0; i < N; ++i) {
      if (affine) {
        dst_pack.elem[i] = static_cast<DST>(src[i]) * weight_pack.elem[i];
      } else {
        dst_pack.elem[i] = static_cast<DST>(src[i]);
      }
    }
    *(reinterpret_cast<layer_norm::PackType<DST, N>*>(dst) + offset) = dst_pack.storage;
  }

  DST* dst;
  const DST* weight;
  int32_t row_size;
};

template <typename SRC, typename DST, bool affine>
struct AffineLoad {
  AffineLoad(const SRC* src, const SRC* weight, int32_t row_size) : src(src), weight(weight), row_size(row_size) {}

  template <int N>
  __device__ void load(DST* dst, int32_t row, int32_t col) const {
    layer_norm::Pack<SRC, N> src_pack;
    layer_norm::Pack<SRC, N> weight_pack;
    const int32_t offset = (row * row_size + col) / N;
    src_pack.storage = *(reinterpret_cast<const layer_norm::PackType<SRC, N>*>(src) + offset);
    if (affine) {
      const int32_t weight_offset = col / N;
      weight_pack.storage = *(reinterpret_cast<const layer_norm::PackType<SRC, N>*>(weight) + weight_offset);
    }
#pragma unroll
    for (int i = 0; i < N; ++i) {
      if (affine) {
        dst[i] = static_cast<DST>(src_pack.elem[i] * weight_pack.elem[i]);
      } else {
        dst[i] = static_cast<DST>(src_pack.elem[i]);
      }
    }
  }
  const SRC* src;
  const SRC* weight;
  int32_t row_size;
};

template <typename T, typename ComputeType, bool affine>
void DispatchRmsNormForwardAffine(cudaStream_t stream, const int64_t nrow, const int64_t ncol, const double eps,
                                  const T* x_dptr, const T* w_dptr, T* y_dptr, ComputeType* inv_rms) {
  layer_norm::DirectLoad<T, ComputeType> load(x_dptr, ncol);
  AffineStore<ComputeType, T, affine> store(y_dptr, w_dptr, ncol);
  CUDA_CHECK(
      (LaunchRmsNorm<decltype(load), decltype(store), ComputeType>(stream, load, store, nrow, ncol, eps, inv_rms)));
}

template <typename T, typename ComputeType>
void RmsNormForward(cudaStream_t stream, const int64_t nrow, const int64_t ncol, const double eps, const T* x_dptr,
                    const T* w_dptr, T* y_dptr, ComputeType* inv_rms) {
  if (w_dptr) {
    DispatchRmsNormForwardAffine<T, ComputeType, true>(stream, nrow, ncol, eps, x_dptr, w_dptr, y_dptr, inv_rms);
  } else {
    DispatchRmsNormForwardAffine<T, ComputeType, false>(stream, nrow, ncol, eps, x_dptr, w_dptr, y_dptr, inv_rms);
  }
}

template <typename T, typename ComputeType, bool affine>
void DispatchRmsNormBackwardAffine(cudaStream_t stream, const int64_t nrow, const int64_t ncol, const T* dy_dptr,
                                   const T* x_dptr, const T* weight_dptr, const ComputeType* inv_rms, T* dx_ptr) {
  layer_norm::DirectLoad<T, ComputeType> load_x(x_dptr, ncol);
  AffineLoad<T, ComputeType, affine> load_dy(dy_dptr, weight_dptr, ncol);
  layer_norm::DirectStore<ComputeType, T> store(dx_ptr, ncol);
  CUDA_CHECK((rms_norm::LaunchRmsNormGrad(stream, nrow, ncol, load_x, load_dy, store, inv_rms)));
}

template <typename T, typename ComputeType>
void RmsNormBackward(cudaStream_t stream, const int64_t nrow, const int64_t ncol, const T* dy_dptr, const T* x_dptr,
                     const T* weight_dptr, const ComputeType* inv_rms, T* dx_dptr) {
  if (weight_dptr) {
    DispatchRmsNormBackwardAffine<T, ComputeType, true>(stream, nrow, ncol, dy_dptr, x_dptr, weight_dptr, inv_rms,
                                                        dx_dptr);
  } else {
    DispatchRmsNormBackwardAffine<T, ComputeType, false>(stream, nrow, ncol, dy_dptr, x_dptr, weight_dptr, inv_rms,
                                                         dx_dptr);
  }
}

template <typename T>
void cuda_rms_norm_tensor_template(CTensorView x, CTensorView weight, CShapeView normalized_shape, float eps,
                                   cudaStream_t stream, CTensorView inv_rms, CTensorView y) {
  const T* weight_dptr = reinterpret_cast<const T*>(weight.ptr);
  const T* x_ptr = reinterpret_cast<const T*>(x.ptr);
  T* y_ptr = reinterpret_cast<T*>(y.ptr);
  using ComputeType = typename layer_norm::DefaultComputeType<T>::type;
  ComputeType* inv_rms_ptr = reinterpret_cast<ComputeType*>(inv_rms.ptr);
  //   const Shape& normalized_shape = ctx->Attr<Shape>("normalized_shape");
  //   const int64_t ncol = normalized_shape.elem_cnt();
  //   const int64_t nrow = inv_rms->shape_view().elem_cnt();
  int64_t ncol = 1;
  for (int i = 0; i < normalized_shape.ndim; i++) {
    ncol *= normalized_shape.shape[i];
  }
  int64_t nrow = 1;
  for (int i = 0; i < inv_rms.ndim; i++) {
    nrow *= inv_rms.shape[i];
  }
  RmsNormForward<T>(stream, nrow, ncol, eps, x_ptr, weight_dptr, y_ptr, inv_rms_ptr);
}

}  // namespace rms_norm
}  // namespace cuda
}  // namespace oneflow

extern "C" {
void cuda_oneflow_rms_norm(CTensorView x, CTensorView weight, CShapeView normalized_shape, float epsilon,
                           cudaStream_t stream, CTensorView inv_rms, CTensorView y) {
  //   const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
  // user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
  // user_op::Tensor* inv_rms = ctx->Tensor4ArgNameAndIndex("inv_rms", 0);
  // const double eps = ctx->Attr<float>("epsilon");
  // const Shape& normalized_shape = ctx->Attr<Shape>("normalized_shape");
  // const int64_t ncol = normalized_shape.elem_cnt();
  // const int64_t nrow = inv_rms->shape_view().elem_cnt();
  // const T* weight_dptr = nullptr;
  // if (ctx->has_input("weight", 0)) {
  //   const auto* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
  //   CHECK_EQ(weight->shape_view().elem_cnt(), ncol);
  //   weight_dptr = weight->dptr<T>();
  // }
  // CHECK_EQ(x->shape_view().elem_cnt(), ncol * nrow);
  // CHECK_LT(nrow * ncol, std::numeric_limits<int32_t>::max())
  //     << "The size of tensor exceeds int32 max limit. The kernel don't support large tensor.";
  // using ComputeType = typename layer_norm::DefaultComputeType<T>::type;
  // rms_norm::RmsNormForward<T>(ctx->stream(), nrow, ncol, eps, x->dptr<T>(), weight_dptr,
  //                             y->mut_dptr<T>(), inv_rms->mut_dptr<ComputeType>());

  switch (x.dtype) {
    case ScalarType::DATA_F16: {
      oneflow::cuda::rms_norm::cuda_rms_norm_tensor_template<half>(x, weight, normalized_shape, epsilon, stream,
                                                                   inv_rms, y);
      break;
    }
    case ScalarType::DATA_BF16: {
      oneflow::cuda::rms_norm::cuda_rms_norm_tensor_template<__nv_bfloat16>(x, weight, normalized_shape, epsilon,
                                                                            stream, inv_rms, y);
      break;
    }
    case ScalarType::DATA_F32: {
      oneflow::cuda::rms_norm::cuda_rms_norm_tensor_template<float>(x, weight, normalized_shape, epsilon, stream,
                                                                    inv_rms, y);
      break;
    }
    case ScalarType::DATA_F64: {
      oneflow::cuda::rms_norm::cuda_rms_norm_tensor_template<double>(x, weight, normalized_shape, epsilon, stream,
                                                                     inv_rms, y);
      break;
    }

    default: {
      throw new std::runtime_error("not supported dtype for cuda_rms_norm_tensor");
    }
  }
}
}