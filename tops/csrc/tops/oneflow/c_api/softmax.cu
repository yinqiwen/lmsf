/*
** BSD 3-Clause License
**
** Copyright (c) 2023, qiyingwang <qiyingwang@tencent.com>, the respective
*contributors, as shown by the AUTHORS file.
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
** IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*ARE
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
#include <stdexcept>

#include "tops/c_api/c_api.h"
#include "tops/oneflow/kernel/softmax.cuh"

template <typename T>
void cuda_tensor_softmax_template(CTensorView input, cudaStream_t cuda_stream,
                                  CTensorView output) {
  int cols = input.shape[input.ndim - 1];
  int rows = 1;
  for (int i = 0; i < input.ndim - 1; i++) {
    rows *= input.shape[i];
  }

  using ComputeType =
      typename oneflow::cuda::softmax::DefaultComputeType<T>::type;
  oneflow::cuda::softmax::DirectLoad<T, ComputeType> load(
      reinterpret_cast<T *>(input.ptr), cols);
  oneflow::cuda::softmax::DirectStore<ComputeType, T> store(
      reinterpret_cast<T *>(output.ptr), cols);
  oneflow::cuda::softmax::DispatchSoftmax<decltype(load), decltype(store),
                                          ComputeType>(cuda_stream, load, store,
                                                       rows, cols);
}

template <typename T>
void cuda_tensor_log_softmax_template(CTensorView input,
                                      cudaStream_t cuda_stream,
                                      CTensorView output) {
  int cols = input.shape[input.ndim - 1];
  int rows = 1;
  for (int i = 0; i < input.ndim - 1; i++) {
    rows *= input.shape[i];
  }
  using ComputeType =
      typename oneflow::cuda::softmax::DefaultComputeType<T>::type;
  oneflow::cuda::softmax::DirectLoad<T, ComputeType> load(
      reinterpret_cast<T *>(input.ptr), cols);
  oneflow::cuda::softmax::DirectStore<ComputeType, T> store(
      reinterpret_cast<T *>(output.ptr), cols);
  oneflow::cuda::softmax::DispatchLogSoftmax<decltype(load), decltype(store),
                                             ComputeType>(cuda_stream, load,
                                                          store, rows, cols);
}

extern "C" {

static void cuda_tensor_softmax_impl(CTensorView input,
                                     cudaStream_t cuda_stream,
                                     CTensorView output) {
  switch (input.dtype) {
  case ScalarType::DATA_F16: {
    cuda_tensor_softmax_template<half>(input, cuda_stream, output);
    break;
  }
  case ScalarType::DATA_BF16: {
    cuda_tensor_softmax_template<__nv_bfloat16>(input, cuda_stream, output);
    break;
  }
  case ScalarType::DATA_F32: {
    cuda_tensor_softmax_template<float>(input, cuda_stream, output);
    break;
  }
  case ScalarType::DATA_F64: {
    cuda_tensor_softmax_template<double>(input, cuda_stream, output);
    break;
  }

  default: {
    throw new std::runtime_error("not supported dtype for cuda_sort_tensor");
  }
  }
}

static void cuda_tensor_log_softmax_impl(CTensorView input,
                                         cudaStream_t cuda_stream,
                                         CTensorView output) {
  switch (input.dtype) {
  case ScalarType::DATA_F16: {
    cuda_tensor_log_softmax_template<half>(input, cuda_stream, output);
    break;
  }
  case ScalarType::DATA_BF16: {
    cuda_tensor_log_softmax_template<__nv_bfloat16>(input, cuda_stream, output);
    break;
  }
  case ScalarType::DATA_F32: {
    cuda_tensor_log_softmax_template<float>(input, cuda_stream, output);
    break;
  }
  case ScalarType::DATA_F64: {
    cuda_tensor_log_softmax_template<double>(input, cuda_stream, output);
    break;
  }

  default: {
    throw new std::runtime_error("not supported dtype for cuda_sort_tensor");
  }
  }
}

void cuda_softmax_tensor(CTensorView input, int algorithm, cudaStream_t stream,
                         CTensorView output) {
  // using ComputeType = typename cuda::softmax::DefaultComputeType<T>::type;
  // oneflow::cuda::softmax::DirectLoad<T, ComputeType> load(x, cols);
  // oneflow::cuda::softmax::DirectStore<ComputeType, T> store(y, cols);
  if (algorithm == 0) {
    cuda_tensor_softmax_impl(input, stream, output);
    // OF_CUDA_CHECK((cuda::softmax::DispatchSoftmax<decltype(load),
    // decltype(store), ComputeType>(cuda_stream, load,
    //                                                                                             store, rows, cols)));
  } else if (algorithm == 1) {
    // OF_CUDA_CHECK((cuda::softmax::DispatchLogSoftmax<decltype(load),
    // decltype(store), ComputeType>(cuda_stream, load,
    //                                                                                                store, rows,
    //                                                                                                cols)));
    cuda_tensor_log_softmax_impl(input, stream, output);
  } else {
    throw new std::runtime_error("invalid algo to do softmax");
  }
}
}

// if there is dim args, this is the dim preprocess

// class SoftmaxFunctorBase {
//  public:
//   Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
//                            const Optional<int64_t>& dim) const {
//     const auto input_shape = input->shape();
//     const int64_t num_axes = input_shape->NumAxes();

//     const auto get_dim = [num_axes]() -> int64_t {
//       const int64_t ndim = num_axes;
//       if (ndim == 0 || ndim == 1 || ndim == 3) {
//         return 0;
//       } else {
//         return 1;
//       }
//     };

//     int64_t dim_ = dim ? JUST(dim) : get_dim();
//     dim_ = JUST(maybe_wrap_dim(dim_, num_axes));
//     if (dim_ != num_axes - 1) {
//       std::vector<int> input_perm(input_shape->dim_vec().size(), 0);
//       for (size_t i = 1; i < input_perm.size(); ++i) { input_perm[i] = i; }
//       input_perm[dim_] = input_perm[input_perm.size() - 1];
//       input_perm[input_perm.size() - 1] = dim_;

//       return sequence_function(functional::Transpose)
//           .then([&](const std::shared_ptr<one::Tensor>& x) {
//             return OpInterpUtil::Dispatch<Tensor>(*op_, {x});
//           })
//           .then(std::bind(functional::Transpose, std::placeholders::_1,
//           input_perm)) .call(input, input_perm);
//     }

//     return OpInterpUtil::Dispatch<Tensor>(*op_, {input});
//   }

//  protected:
//   SoftmaxFunctorBase() = default;
//   virtual ~SoftmaxFunctorBase() = default;

//   std::shared_ptr<OpExpr> op_;
// };