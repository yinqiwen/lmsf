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
#include <string.h>
#include "tops/c_api/c_api.h"

namespace tops {
class ShapeView {
 public:
  ShapeView(const CTensorView& tensor) {
    shapes_.ndim = tensor.ndim;
    memcpy(shapes_.shape, tensor.shape, sizeof(int64_t) * 4);
    for (int i = 0; i < shapes_.ndim; i++) {
      data_[i] = shapes_.shape[i];
    }
  }
  int64_t elem_cnt() const {
    int64_t n = 1;
    for (int i = 0; i < shapes_.ndim; i++) {
      n *= shapes_.shape[i];
    }
    return n;
  }
  int64_t At(int64_t index) const { return shapes_.shape[index]; }
  int64_t Count(int64_t begin_axis, int64_t end_axis) const {
    int64_t cnt = 1;
    for (int64_t i = begin_axis; i < end_axis; ++i) {
      cnt *= At(i);
    }
    return cnt;
  }

  int64_t Count(int64_t begin_axis) const { return Count(0, shapes_.ndim); }
  int64_t ndims() const { return shapes_.ndim; }
  int64_t NumAxes() const { return shapes_.ndim; }

  const int32_t* data() const { return data_; }

 private:
  CShapeView shapes_;
  int32_t data_[4];
};
}  // namespace tops