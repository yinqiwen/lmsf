/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// #include "oneflow/core/framework/framework.h"
// #include "oneflow/user/kernels/dim_gather_kernel_util.h"
#include "tops/oneflow/common/cuda_util.h"
#include "tops/oneflow/common/nd_index_offset_helper.h"
namespace oneflow {
constexpr int kDimGatherMaxDimCount = 4;

template <typename T>
using DimOpIndexNdHelper = NdIndexOffsetHelper<T, kDimGatherMaxDimCount>;

template <typename IN_T, typename IDX_T>
OF_DEVICE_FUNC void DoDimGather(const DimOpIndexNdHelper<IDX_T>& input_nd_helper,
                                const DimOpIndexNdHelper<IDX_T>& index_nd_helper, int ndim, int64_t elem_cnt,
                                int32_t dim_length, int32_t dim, const IDX_T* index, const IN_T* input, IN_T* output) {
  CUDA_1D_KERNEL_LOOP(index_offset, elem_cnt) {
    IDX_T coordinate[kDimGatherMaxDimCount] = {0};
    const IDX_T x = index[index_offset];
#ifdef __CUDA_ARCH__
    assert(x < dim_length && "gather index is out of bounds");
#else
    // CHECK_LE(x, dim_length) << "RuntimeError: index " << x << " is out of bounds for dimension " << dim << " with
    // size "
    //                         << dim_length;
#endif
    index_nd_helper.OffsetToNdIndex(index_offset, coordinate, ndim);
    coordinate[dim] = x;

    IDX_T input_offset = input_nd_helper.NdIndexToOffset(coordinate, ndim);
    output[index_offset] = input[input_offset];
  }
}

namespace user_op {

template <typename IN_T, typename IDX_T>
__global__ void DoCUDADimGather(const DimOpIndexNdHelper<IDX_T> input_nd_helper,
                                const DimOpIndexNdHelper<IDX_T> index_nd_helper, int ndim, int64_t elem_cnt,
                                int32_t dim_length, int32_t dim, const IDX_T* index, const IN_T* input, IN_T* output) {
  DoDimGather<IN_T, IDX_T>(input_nd_helper, index_nd_helper, ndim, elem_cnt, dim_length, dim, index, input, output);
}

template <typename IN_T, typename IDX_T>
struct DimGatherFunctor final {
  void operator()(cudaStream_t stream, const DimOpIndexNdHelper<IDX_T>& input_nd_helper,
                  const DimOpIndexNdHelper<IDX_T>& index_nd_helper, int ndim, int64_t elem_cnt, int32_t dim_length,
                  int32_t dim, const IDX_T* index, const IN_T* input, IN_T* output) {
    RUN_CUDA_KERNEL((DoCUDADimGather<IN_T, IDX_T>), stream, BlocksNum4ThreadsNum(elem_cnt), input_nd_helper,
                    index_nd_helper, ndim, elem_cnt, dim_length, dim, index, input, output);
  }
};

// float16 special case of DimGatherFunctor template
template <typename IDX_T>
struct DimGatherFunctor<float16, IDX_T> final {
  void operator()(cudaStream_t stream, const DimOpIndexNdHelper<IDX_T>& input_nd_helper,
                  const DimOpIndexNdHelper<IDX_T>& index_nd_helper, int ndim, int64_t elem_cnt, int32_t dim_length,
                  int32_t dim, const IDX_T* index, const float16* input, float16* output) {
    RUN_CUDA_KERNEL((DoCUDADimGather<half, IDX_T>), stream, BlocksNum4ThreadsNum(elem_cnt), input_nd_helper,
                    index_nd_helper, ndim, elem_cnt, dim_length, dim, index, reinterpret_cast<const half*>(input),
                    reinterpret_cast<half*>(output));
  }
};

// OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_DIM_GATHER_FUNCTOR, (DeviceType::kCUDA),
//                                  DIM_GATHER_SCATTER_DATA_TYPE_CUDA_SEQ, INDEX_DATA_TYPE_SEQ);

}  // namespace user_op
}  // namespace oneflow
