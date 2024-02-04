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

#pragma once
#include <cstring>
#include <vector>
#include "tops/c_api/c_api.h"

namespace at::native {
using DimVector = std::vector<int64_t>;
}

extern "C" {

inline at::native::DimVector tensor_get_shape(const CTensorView& t) {
  at::native::DimVector s(t.ndim);
  for (int i = 0; i < t.ndim; i++) {
    s[i] = t.shape[i];
  }
  return s;
}
inline at::native::DimVector tensor_get_stride(const CTensorView& t) {
  at::native::DimVector s(t.ndim);
  for (int i = 0; i < t.ndim; i++) {
    s[i] = t.stride[i];
  }
  return s;
}

inline CTensorView restride_dim(const CTensorView& t, int64_t dim, const int64_t* replacement_shape) {
  CTensorView result = t;
  result.stride[dim] = 0;
  memcpy(result.shape, replacement_shape, sizeof(int64_t) * 4);
  return result;
}

inline CTensorView tensor_as_strided(const CTensorView& t, const int64_t* replacement_shape,
                                     const int64_t* replacement_strides) {
  CTensorView result = t;
  memcpy(result.shape, replacement_shape, sizeof(int64_t) * 4);
  memcpy(result.stride, replacement_strides, sizeof(int64_t) * 4);
  return result;
}
}