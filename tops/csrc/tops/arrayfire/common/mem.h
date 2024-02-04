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

#pragma once
#include "tops/arrayfire/common/array.h"
#include "tops/arrayfire/common/cuda_ptr.h"
#include "tops/arrayfire/common/dim4.h"
#include "tops/arrayfire/common/param.h"
#include "tops/common/err_cuda.h"
#include <cuda_runtime.h>
#include <memory>
#include <vector>

namespace arrayfire {
template <typename T> cuda_unique_ptr<T> device_alloc(size_t n) {
  T *t = nullptr;
  CUDA_CHECK(cudaMalloc(&t, sizeof(T) * n));
  cuda_unique_ptr<T> p(t);
  return p;
}

template <typename T> arrayfire::cuda::Array<T> createEmptyArray(dim4 dims) {
  size_t n = dims.elements();
  cuda_unique_ptr<T> p = device_alloc<T>(n);
  dim4 strides = calcStrides(dims);
  return arrayfire::cuda::Array<T>(std::move(p), dims, strides);
}

template <typename T>
arrayfire::cuda::Array<T> createArray(const std::vector<T> &data, dim4 dims,
                                      cudaStream_t stream) {
  size_t n = dims.elements();
  cuda_unique_ptr<T> p = device_alloc<T>(n);
  CUDA_CHECK(cudaMemcpyAsync(p.get(), data.data(), n * sizeof(T),
                             cudaMemcpyKind::cudaMemcpyHostToDevice, stream));
  dim4 strides = calcStrides(dims);
  return arrayfire::cuda::Array<T>(std::move(p), dims, strides);
}

template <typename T>
arrayfire::cuda::Array<T> copyArray(const arrayfire::cuda::Array<T> &in,
                                    cudaStream_t stream) {
  auto dims = in.dims();
  size_t n = dims.elements();
  cuda_unique_ptr<T> p = device_alloc<T>(n);
  CUDA_CHECK(cudaMemcpyAsync(p.get(), in.get(), n * sizeof(T),
                             cudaMemcpyKind::cudaMemcpyDeviceToDevice, stream));
  dim4 strides = calcStrides(dims);
  return arrayfire::cuda::Array<T>(std::move(p), dims, strides);
}

template <typename T>
arrayfire::cuda::Array<T> createParamArray(arrayfire::cuda::Param<T> tmp,
                                           bool owner) {
  return arrayfire::cuda::Array<T>(tmp, owner);
}

} // namespace arrayfire