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
#include "tops/torch/common/tensor_iterator.h"
#include <stdio.h>
#include <numeric>
#include <vector>
namespace at::native {

template <typename Container, typename ArrayType>
Container infer_size_dimvector(ArrayType a, ArrayType b) {
  size_t dimsA = a.size();
  size_t dimsB = b.size();
  size_t ndim = dimsA > dimsB ? dimsA : dimsB;
  Container expandedSizes(ndim);

  // Use ptrdiff_t to ensure signed comparison.
  for (ptrdiff_t i = (ptrdiff_t)ndim - 1; i >= 0; --i) {
    ptrdiff_t offset = ndim - 1 - i;
    ptrdiff_t dimA = dimsA - 1 - offset;
    ptrdiff_t dimB = dimsB - 1 - offset;
    auto sizeA = (dimA >= 0) ? a[dimA] : 1;
    auto sizeB = (dimB >= 0) ? b[dimB] : 1;

    // TORCH_CHECK(sizeA == sizeB || sizeA == 1 || sizeB == 1, "The size of tensor a (", sizeA,
    //             ") must match the size of tensor b (", sizeB, ") at non-singleton dimension ", i);

    // 1s map to the other size (even 0).
    expandedSizes[i] = sizeA == 1 ? std::move(sizeB) : std::move(sizeA);
  }

  return expandedSizes;
}

void TensorIterator::permute_dimensions(const std::vector<int64_t>& perm) {
  // TORCH_INTERNAL_ASSERT(perm.size() == static_cast<unsigned>(ndim()));

  auto reorder = [perm](int64_t* data) {
    auto res = DimVector(perm.size(), 0);
    for (int i = 0; i < perm.size(); i++) {
      // for (const auto i : c10::irange(perm.size())) {
      res[i] = data[perm[i]];
    }
    memcpy(data, res.data(), sizeof(int64_t) * res.size());

    // return res;
  };

  // Update shape and strides
  // shape_ = reorder(shape_);
  reorder(shape_.data());
  for (auto& op : operands_) {
    if (op.ndim > 0) {
      reorder(op.stride);
    }
    // if (!op.stride_bytes.empty()) {
    //   op.stride_bytes = reorder(op.stride_bytes);
    // }
  }
}

void TensorIterator::reorder_dimensions() {
  // Sort the dimensions based on strides in ascending order with reduced dims
  // at the front. NOTE: that this inverts the order of C-contiguous tensors.
  // strides[0] is the fastest moving dimension instead of strides[ndim - 1].
  // See NOTE: [Computing output strides] and inline  comments for more detailed
  // description

  perm_.resize(ndim());
  if (ndim() == 1) {
    perm_[0] = 0;
    return;
  }

  // initialize perm with n-1, n-2, ..., 1, 0
  std::iota(perm_.rbegin(), perm_.rend(), 0);

  // Reordering dimensions changes iteraton order
  // if (enforce_linear_iteration_) {
  //   permute_dimensions(perm_);
  //   return;
  // }

  // returns 1 if the dim0 should come after dim1, -1 if dim0 should come
  // before dim1, and 0 if the comparison is ambiguous.
  auto should_swap = [&](size_t dim0, size_t dim1) {
    // for (const auto arg : c10::irange(ntensors())) {
    for (int arg = 0; arg < operands_.size(); arg++) {
      // ignore undefined or incorrectly sized tensors
      // if (operands_[arg].stride_bytes.empty() || operands_[arg].will_resize) {
      if (operands_[arg].ndim == 0) {
        continue;
      }
      int64_t stride0 = operands_[arg].stride[dim0];
      int64_t stride1 = operands_[arg].stride[dim1];
      // if (is_reduction_ && operands_[arg].is_output) {
      if (is_reduction_ && arg == 0) {
        // move reduced dimensions to the front
        // strides of reduced dimensions are always set to 0 by
        // review_reduce_result
        if ((stride0 == 0) != (stride1 == 0)) {
          return stride1 == 0 ? 1 : -1;
        }
      }
      // move on to the next input if one of the dimensions is broadcasted
      if (stride0 == 0 || stride1 == 0) {
        continue;
        // it is important to return here only with strict comparisons, for
        // equal strides we try to break the tie later by comparing
        // corresponding dimensions or if that does not work, moving on to the
        // next tensor
      } else if (stride0 < stride1) {
        return -1;
      } else if (stride0 > stride1) {
        return 1;
      } else {  // equal strides, use dimensions themselves as the tie-breaker.
        // at this point, with zero strides out of the way, we are guaranteed
        // that operand dimensions are equal to shape_
        auto t_dim0 = shape_[dim0];
        auto t_dim1 = shape_[dim1];
        // return only if dimensions should be swapped, otherwise move on to the
        // next tensor
        if (t_dim0 > t_dim1) {
          return 1;
        }
      }
    }
    return 0;
  };

  // insertion sort with support for ambiguous comparisons
  // for (const auto i : c10::irange(1, ndim())) {
  for (int i = 1; i < ndim(); i++) {
    int dim1 = i;
    for (int dim0 = i - 1; dim0 >= 0; dim0--) {
      int comparison = should_swap(perm_[dim0], perm_[dim1]);
      if (comparison > 0) {
        std::swap(perm_[dim0], perm_[dim1]);
        dim1 = dim0;
      } else if (comparison < 0) {
        break;
      }
    }
  }

  // perform re-ordering of shape and strides
  permute_dimensions(perm_);
}

void TensorIterator::coalesce_dimensions() {
  if (ndim() <= 1) {
    return;
  }

  // We can coalesce two adjacent dimensions if either dim has size 1 or if:
  // shape[n] * stride[n] == stride[n + 1].
  auto can_coalesce = [&](int dim0, int dim1) {
    auto shape0 = shape_[dim0];
    auto shape1 = shape_[dim1];
    if (shape0 == 1 || shape1 == 1) {
      return true;
    }
    for (int i = 0; i < operands_.size(); i++) {
      auto stride = operands_[i].stride;
      if (shape0 * stride[dim0] != stride[dim1]) {
        return false;
      }
    }
    return true;
  };

  // replace each operands stride at dim0 with its stride at dim1
  auto replace_stride = [&](int dim0, int dim1) {
    for (int i = 0; i < operands_.size(); i++) {
      auto stride = operands_[i].stride;
      stride[dim0] = stride[dim1];
    }
  };

  int prev_dim = 0;
  // for (const auto dim : c10::irange(1, ndim())) {
  for (int dim = 1; dim < ndim(); dim++) {
    if (can_coalesce(prev_dim, dim)) {
      if (shape_[prev_dim] == 1) {
        replace_stride(prev_dim, dim);
      }
      shape_[prev_dim] *= shape_[dim];
    } else {
      prev_dim++;
      if (prev_dim != dim) {
        replace_stride(prev_dim, dim);
        shape_[prev_dim] = shape_[dim];
      }
    }
  }

  shape_.resize(prev_dim + 1);
  for (int i = 0; i < operands_.size(); i++) {
    // operands_[i].stride_bytes.resize(ndim());
  }
  has_coalesced_dimensions_ = true;

  // std::cout << "coalesce_dimensions shape:" << shape_ << std::endl;
}

void TensorIterator::compute_strides() {
  for (auto& op : operands_) {
    // IntArrayRef original_shape = config.static_shape_ ? shape_ : op.tensor_base().sizes();
    auto original_shape = tensor_get_shape(op);
    auto original_stride = tensor_get_stride(op);
    auto element_size_in_bytes = element_size(static_cast<ScalarType>(op.dtype));
    auto offset = ndim() - original_shape.size();
    // if (offset > 0)
    //   op.stride_bytes.resize(ndim(), 0);
    // else
    //   op.stride_bytes.resize(ndim());
    // for (const auto i : c10::irange(original_shape.size())) {
    for (int i = 0; i < original_shape.size(); i++) {
      // see NOTE: [Computing output strides]
      if (original_shape[i] == 1 && shape_[offset + i] != 1) {
        op.stride[offset + i] = 0;
      } else {
        op.stride[offset + i] = original_stride[i] * element_size_in_bytes;
      }
    }
  }
}

void TensorIterator::compute_shape() {
  all_ops_same_shape_ = true;
  bool has_scalars = false;
  bool has_tensors = false;
  bool is_all_tensor_continous = true;
  for (auto& op : operands_) {
    // if (!op.tensor_base().defined()) continue;

    // For now, don't include output tensors when we're resizing outputs.
    // These shapes don't participate in shape computation.
    // This preserves the legacy behavior where torch.add(..., out=dst) resizes
    // the destination tensor.  If the output tensor is also an input, we'll
    // pick it up later in the operands.
    // if (config.resize_outputs_ && op.is_output) continue;
    // TORCH_CHECK(!op.tensor_base().unsafeGetTensorImpl()->has_symbolic_sizes_strides(),
    //             "TensorIterator does not support symbolic shapes; please implement this operator in torch/_refs "
    //             "using the elementwise or reduction helpers (look at backtrace to find out what operator this is)");
    auto shape = tensor_get_shape(op);
    if (shape.empty()) {
      has_scalars = true;
    } else {
      has_tensors = true;
    }
    if (has_scalars && has_tensors) {
      all_ops_same_shape_ = false;
    }
    if (shape_.empty()) {
      shape_ = shape;
    } else if (shape != shape_) {
      all_ops_same_shape_ = false;
      shape_ = infer_size_dimvector<DimVector, DimVector>(shape_, shape);
    }

    is_all_tensor_continous &= is_tensor_contiguous(&op);
  }
  all_ops_are_scalars_ = !has_tensors;

  if (is_all_tensor_continous) {
    int64_t numel = this->numel();
    shape_[0] = numel;
    shape_.resize(1);

    for (auto& op : operands_) {
      auto element_size_in_bytes = element_size(static_cast<ScalarType>(op.dtype));
      // op.stride_bytes.resize(ndim());
      // printf("###element_size_in_bytes:%d\n", element_size_in_bytes);
      if (ndim() > 0) {
        op.stride[0] = element_size_in_bytes;
      }
    }
  } else {
    compute_strides();
    reorder_dimensions();
    coalesce_dimensions();
  }
}

}  // namespace at::native