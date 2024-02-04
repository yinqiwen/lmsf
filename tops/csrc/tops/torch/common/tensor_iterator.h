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
#include <vector>
#include "tops/c_api/c_api.h"
#include "tops/torch/common/tensor.h"

namespace at::native {
#define C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE 5
constexpr size_t kDimVectorStaticSize = C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE;
/// A container for sizes or strides
class TensorIterator {
 private:
  std::vector<CTensorView> operands_;
  std::vector<int64_t> perm_;
  DimVector shape_;
  bool all_ops_same_shape_ = false;
  bool all_ops_are_scalars_ = false;
  bool is_reduction_ = false;
  bool has_coalesced_dimensions_ = false;

  void compute_shape();
  void compute_strides();
  void reorder_dimensions();
  void permute_dimensions(const std::vector<int64_t>& perm);
  void coalesce_dimensions();

 public:
  TensorIterator(const std::vector<CTensorView>& ts) : operands_(ts) { compute_shape(); }
  void* get_ptr(int i) const { return operands_[i].ptr; }
  int ndim() const { return shape_.size(); }
  const DimVector& shape() const { return shape_; }
  const int64_t* strides(int i) const { return operands_[i].stride; }
  int64_t numel() const {
    int64_t numel = 1;
    for (int64_t size : shape_) {
      numel *= size;
    }
    return numel;
  }
};
}  // namespace at::native