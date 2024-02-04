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
#include "tops/c_api/c_api.h"
#include <string>
#include <vector>
extern "C" {

cudaDeviceProp *getCudaDeviceProp() {
  static cudaDeviceProp *default_props = nullptr;
  static cudaDeviceProp props;
  if (nullptr == default_props) {
    cudaGetDeviceProperties(&props, 0);
    default_props = &props;
  }
  return default_props;
}
size_t get_tensor_element_count(const CTensorView *tensor) {
  return tensor->shape[0] * tensor->shape[1] * tensor->shape[2] * tensor->shape[3];
}
size_t element_size(ScalarType type) {
  switch (type) {
    case ScalarType::DATA_U8: {
      return 1;
    }
    case ScalarType::DATA_BF16:
    case ScalarType::DATA_F16: {
      return 2;
    }
    case ScalarType::DATA_U32:
    case ScalarType::DATA_F32: {
      return 4;
    }
    case ScalarType::DATA_I64:
    case ScalarType::DATA_F64: {
      return 8;
    }
    default: {
      return 0;
    }
  }
}

bool is_tensor_contiguous(const CTensorView *t) {
  if (t->ndim == 1) {
    return true;
  }
  if (t->stride[t->ndim - 1] != 1) {
    return false;
  }
  // todo
  return true;
}
}