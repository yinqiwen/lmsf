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
#include <cuda_runtime_api.h>
#include <cstdint>
#include <vector>
#include "tops/c_api/c_api.h"
#include "tops/torch/common/non_empty_utils.h"
#include "tops/torch/common/tensor.h"
#include "tops/torch/common/tensor_iterator.h"
#include "tops/torch/kernel/offset_calculator.cuh"
#include "tops/torch/kernel/thread_constants.h"

namespace at::native {
class TensorAssign {
 public:
  template <typename scalar_t>
  constexpr __device__ void operator()(scalar_t* self_data_start, int64_t index, int64_t numel,
                                       const scalar_t* src_data) const {
    (void)numel;  // suppress unused warning
    // printf("assign with val:%lld at %p\n", *src_data, self_data_start + index);
    *(self_data_start + index) = *src_data;
    // printf("after assign val:%lld \n", *(self_data_start + index));
  }
};
static TensorAssign tensor_assign;

template <int nt, int vt, typename func_t>
__global__ void _scatter_gather_elementwise_kernel(int N, func_t f) {
  constexpr int nv = nt * vt;
  int idx = nv * blockIdx.x + threadIdx.x;

#pragma unroll
  for (int i = 0; i < vt; ++i) {
    if (idx < N) {
      f(idx);
      idx += nt;
    }
  }
}
template <int nt, int vt, typename func_t>
static void _launch_scatter_gather_kernel(int64_t N, const func_t& f, cudaStream_t stream) {
  //   TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }

  const dim3 block(nt);
  const dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  //   const auto stream = at::cuda::getCurrentCUDAStream();
  _scatter_gather_elementwise_kernel<nt, vt, func_t><<<grid, block, 0, stream>>>(N, f);
  //   C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool is_scatter_like, typename scalar_t, typename index_scalar_t>
struct _cuda_scatter_gather_internal_kernel {
  template <typename func_t>
  void operator()(
      const TensorIterator& iter, int64_t index_size, int64_t index_stride,
      int64_t numel,  // Do not use `const` qualifier here as it may cause issue in cuda 11.6.x. See #75434, #75545
      const func_t& f, cudaStream_t stream) {
    // if (!iter.can_use_32bit_indexing()) {
    //   for (auto& sub_iter : iter.with_32bit_indexing()) {
    //     _cuda_scatter_gather_internal_kernel<is_scatter_like, scalar_t>()(sub_iter, index_size, index_stride, numel,
    //     f);
    //   }
    //   return;
    // }

    char* self_ptr = (char*)iter.get_ptr(0);
    char* src_ptr = (char*)iter.get_ptr(1);
    char* index_ptr = (char*)iter.get_ptr(2);

    auto offset_calc = make_offset_calculator<3>(iter);
    auto loop = [=] __device__(int i) {
      auto offsets = offset_calc.get(i);
      int64_t idx_dim = *(index_scalar_t*)(index_ptr + offsets[2]);
      // printf("###get %d %d/%d/%d %lld index_stride:%lld\n", i, offsets[0], offsets[1], offsets[2], idx_dim,
      //        index_stride);
      f((scalar_t*)(self_ptr + offsets[0]), is_scatter_like ? idx_dim * index_stride : 0, numel,
        (scalar_t*)(src_ptr + offsets[1]) + (is_scatter_like ? 0 : idx_dim * index_stride));
    };
    // int64_t tensor_numel = 4;
    _launch_scatter_gather_kernel<num_threads(), thread_work_size()>(iter.numel(), loop, stream);
  }
};  // struct _cuda_scatter_fill_internal_kernel

template <typename DTYPE, typename index_scalar_t, bool is_scatter_like = true, bool cast_to_opaque = true>
struct cuda_scatter_gather_base_kernel {
  void operator()(const CTensorView& self, int64_t dim, const CTensorView& index, const CTensorView& src,
                  const TensorAssign& f, cudaStream_t stream) {
    // at::assert_no_internal_overlap(self);

    // auto index_sizes = ensure_nonempty_vec(index.sizes().vec());
    // auto self_strides = ensure_nonempty_vec(self.strides().vec());
    // auto src_strides = ensure_nonempty_vec(src.strides().vec());
    auto self_restrided = restride_dim(self, dim, index.shape);
    auto src_restrided = tensor_as_strided(src, index.shape, src.stride);
    // printf("self shape %lld/%lld, stride: %lld/%lld\n", self_restrided.shape[0], self_restrided.shape[1],
    //        self_restrided.stride[0], self_restrided.stride[1]);
    // printf("src shape %lld/%lld, stride: %lld/%lld\n", src_restrided.shape[0], src_restrided.shape[1],
    //        src_restrided.stride[0], src_restrided.stride[1]);

    // restride self and src such that
    // self.shape = src.shape = index.shape
    //
    // restride stride[dim] such that
    // if (is_scatter_like) self.stride[dim] = 0
    // else src.stride[dim] = 0
    // auto self_restrided =
    //     is_scatter_like ? restride_dim(self, dim, index_sizes) : self.as_strided(index_sizes, self_strides);
    // auto src_restrided =
    //     is_scatter_like ? src.as_strided(index_sizes, src_strides) : restride_dim(src, dim, index_sizes);
    // printf("##self_restrided stride: %lld/%lld\n", self_restrided.stride[0], self_restrided.stride[1]);
    // printf("##src_restrided stride: %lld/%lld\n", src_restrided.stride[0], src_restrided.stride[1]);
    // printf("##index stride: %lld/%lld\n", index.stride[0], index.stride[1]);

    auto self_dim_stride = ensure_nonempty_stride(self, dim);
    auto self_dim_size = ensure_nonempty_size(self, dim);

    auto src_dim_stride = ensure_nonempty_stride(src, dim);
    auto src_dim_size = ensure_nonempty_size(src, dim);

    auto index_size = is_scatter_like ? self_dim_size : src_dim_size;
    auto index_stride = is_scatter_like ? self_dim_stride : src_dim_stride;

    std::vector<CTensorView> ops{
        self_restrided,
        src_restrided,
        index,
    };
    TensorIterator iter(ops);
    // printf("index_size:%lld, index_stride:%lld, numel:%lld\n", index_size, index_stride, iter.numel());
    // printf("##iter.strides(0):%lld\n", *iter.strides(0));
    // printf("##iter.strides(1):%lld\n", *iter.strides(1));
    // printf("##iter.strides(2):%lld\n", *iter.strides(2));
    int64_t numel = self.shape[0] * self.shape[1] * self.shape[2] * self.shape[3];
    _cuda_scatter_gather_internal_kernel<is_scatter_like, DTYPE, index_scalar_t>()(iter, index_size, index_stride,
                                                                                   numel, f, stream);
  }
};

}  // namespace at::native