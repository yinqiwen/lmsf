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
#include "tops/arrayfire/common/cuda_ptr.h"
#include "tops/arrayfire/common/dim4.h"
#include "tops/arrayfire/common/param.h"
#include "tops/c_api/c_api.h"
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace arrayfire {
namespace cuda {
template <typename T> class Array {
public:
  Array() {}

  Array(cuda_unique_ptr<T> &&p, const dim4 &dims, const dim4 &strides)
      : ptr_(std::move(p)), param_(ptr_.get(), dims.dims, strides.dims) {}

  Array(Param<T> param, bool own) : param_(param) {
    CudaObjFree<T> free(own);
    ptr_ = cuda_unique_ptr<T>(param_.ptr, free);
  }

  void reset() { ptr_.reset(); }

  bool is_empty() const { return ptr_ == nullptr; }

  dim4 dims() const {
    return dim4{param_.dims[0], param_.dims[1], param_.dims[2], param_.dims[3]};
  }

  size_t elements() const { return param_.elements(); }

  Param<T> GetKernelParams() { return param_; }

  CParam<T> GetKernelParams() const { return CParam<T>(param_); }

  void setDataDims(const dim4 &dims) {
    dim4 strides = calcStrides(dims);
    param_ = Param<T>(ptr_.get(), dims.dims, strides.dims);
  }

  T *get() { return ptr_.get(); }

  const T *get() const { return ptr_.get(); }

  std::string to_string() const {
    size_t n = dims().elements();
    std::vector<T> data(n);
    cudaMemcpy(data.data(), get(), n * sizeof(T),
               cudaMemcpyKind::cudaMemcpyDeviceToHost);
    std::string str;
    str.append("[");
    for (auto &t : data) {
      str.append(std::to_string(t));
      str.append(",");
    }
    str.append("]");
    return str;
  }

  static Array<T> create(CTensorView view) {
    arrayfire::dim4 dims(view.shape[0], view.shape[1], view.shape[2],
                         view.shape[3]);
    arrayfire::dim4 strides = arrayfire::calcStrides(dims);

    Param<T> param(reinterpret_cast<T *>(view.ptr), dims.dims, strides.dims);
    return Array<T>(param, false);
  }

private:
  cuda_unique_ptr<T> ptr_;
  Param<T> param_;
};
} // namespace cuda
} // namespace arrayfire