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

#include "tops/arrayfire/common/types.h"

namespace arrayfire {
namespace cuda {

template <typename T> class Param {
public:
  dim_t dims[4];
  dim_t strides[4];
  T *ptr;

  __DH__ Param() noexcept : dims(), strides(), ptr(nullptr) {}

  __DH__
  Param(T *iptr, const dim_t *idims, const dim_t *istrides) noexcept
      : dims{idims[0], idims[1], idims[2], idims[3]}, strides{istrides[0],
                                                              istrides[1],
                                                              istrides[2],
                                                              istrides[3]},
        ptr(iptr) {}

  __DH__ size_t elements() const noexcept {
    return dims[0] * dims[1] * dims[2] * dims[3];
  }

  dim_t *dims_ptr() { return dims; }

  dim_t *strides_ptr() { return strides; }

  Param(const Param<T> &other) noexcept = default;
  Param(Param<T> &&other) noexcept = default;
  Param<T> &operator=(const Param<T> &other) noexcept = default;
  Param<T> &operator=(Param<T> &&other) noexcept = default;
};

template <typename T> Param<T> flat(Param<T> in) {
  in.dims[0] = in.elements();
  in.dims[1] = 1;
  in.dims[2] = 1;
  in.dims[3] = 1;
  return in;
}

template <typename T> class CParam {
public:
  dim_t dims[4];
  dim_t strides[4];
  const T *ptr;

  __DH__ CParam(const T *iptr, const dim_t *idims, const dim_t *istrides)
      : dims{idims[0], idims[1], idims[2], idims[3]}, strides{istrides[0],
                                                              istrides[1],
                                                              istrides[2],
                                                              istrides[3]},
        ptr(iptr) {}

  __DH__ CParam(const Param<T> &in)
      : dims{in.dims[0], in.dims[1], in.dims[2], in.dims[3]},
        strides{in.strides[0], in.strides[1], in.strides[2], in.strides[3]},
        ptr(in.ptr) {}

  __DH__ size_t elements() const noexcept {
    return dims[0] * dims[1] * dims[2] * dims[3];
  }

  CParam(const CParam<T> &other) noexcept = default;
  CParam(CParam<T> &&other) noexcept = default;
  CParam<T> &operator=(const CParam<T> &other) noexcept = default;
  CParam<T> &operator=(CParam<T> &&other) noexcept = default;
};

} // namespace cuda
} // namespace arrayfire
