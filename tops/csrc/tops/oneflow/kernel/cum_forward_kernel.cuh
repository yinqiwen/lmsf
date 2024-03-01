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
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <type_traits>
#include "tops/common/shape.h"
#include "tops/oneflow/common/cuda_util.h"
// #include "oneflow/core/device/cuda_util.h"
// #include "oneflow/core/ep/cuda/cuda_stream.h"
// #include "oneflow/core/framework/framework.h"
// #include "oneflow/core/kernel/new_kernel_util.h"
// #include "oneflow/core/ndarray/binary_func.h"

namespace oneflow {

namespace {

template <typename T>
inline T CeilDiv(T n, T m) {
  return (n + m - 1) / m;
}

template <typename T>
struct SumFunctor {
  __device__ __forceinline__ T operator()(const T a, const T b) const { return a + b; }
};
template <typename T>
struct ProdFunctor {
  __device__ __forceinline__ T operator()(const T a, const T b) const { return a * b; }
};

// template <typename T, template <typename> class BinaryFunc>
// size_t InferTmpBufferSize(user_op::InferContext* ctx) {
//   const Shape& in_shape = ctx->InputShape("x", 0);
//   const int64_t dim = ctx->Attr<int64_t>("dim");
//   const size_t dim_size = in_shape.At(dim);
//   if (in_shape.elem_cnt() == dim_size) {
//     size_t temp_storage_bytes = 0;
//     OF_CUDA_CHECK(cub::DeviceScan::InclusiveScan(nullptr, temp_storage_bytes, static_cast<T*>(nullptr),
//                                                  static_cast<T*>(nullptr), BinaryFunc<T>(), dim_size));
//     return GetCudaAlignedSize(temp_storage_bytes);
//   }
//   return 0;
// }

// total thread number: cs_up_space * cs_down_space
// in cs_down_space part, use cs_down_space threads
// to calculate as follows(m=cs_down_space-1, n=cs_space-1, '|' stands for dependency):
template <typename T, template <typename> class BinaryFunc>
__global__ void CumForwardGpu(const T* in_ptr, T* out_ptr, int64_t cs_up_space, int64_t cs_space,
                              int64_t cs_down_space) {
  CUDA_1D_KERNEL_LOOP(i, cs_up_space * cs_down_space) {
    auto cs_up_space_id = i / cs_down_space;
    auto cs_down_space_id = i - (i / cs_down_space) * cs_down_space;

    auto* in_ptr_base = in_ptr + cs_up_space_id * cs_space * cs_down_space + cs_down_space_id;
    auto* out_ptr_base = out_ptr + cs_up_space_id * cs_space * cs_down_space + cs_down_space_id;

    // calculate cs_space data in one thread
    for (auto j = 0; j < cs_space; j++) {
      auto idx = j * cs_down_space;
      out_ptr_base[idx] = in_ptr_base[idx];
      if (j != 0) {
        out_ptr_base[idx] = BinaryFunc<T>()(out_ptr_base[idx], out_ptr_base[idx - cs_down_space]);
      }
    }
  }
}

template <typename T, template <typename> class BinaryFunc>
void ScanOuterDim(cudaDeviceProp* prop, cudaStream_t stream, tops::ShapeView in_shape, int64_t dim, const T* in_ptr,
                  T* out_ptr) {
  // todo
  // data partition: up_space|space|down_space
  auto up_space = in_shape.elem_cnt() / in_shape.Count(dim);
  auto space = in_shape.At(dim);
  auto down_space = in_shape.Count(dim + 1);
  auto thread_num = up_space * down_space;

  oneflow::CudaLaunchConfig launch_config;
  const size_t max_waves = 1;
  static constexpr uint32_t kDefaultBlockSize = 256;
  constexpr uint32_t block_size = kDefaultBlockSize;
  oneflow::InitLaunchConfigWithWaves(&launch_config, thread_num, block_size, max_waves, prop);
  // RUN_CUDA_KERNEL((CumForwardGpu<T, BinaryFunc>), stream, thread_num, in_ptr, out_ptr, up_space, space, down_space);
  CumForwardGpu<T, BinaryFunc>
      <<<launch_config.grid_dim, launch_config.block_dim, launch_config.shared_mem_size, stream>>>(
          in_ptr, out_ptr, up_space, space, down_space);
}

// Refer from
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/ScanKernels.cu
template <typename T, int num_threads_x, int num_threads_y, template <typename> class BinaryFunc>
__device__ void ScanInnerMostDimKernelImpl(T* row_buf, T* src_, T* tgt_, const uint32_t num_rows,
                                           const uint32_t row_size, T init) {
  for (uint32_t block_row = blockIdx.x * blockDim.y; block_row < num_rows; block_row += blockDim.y * gridDim.x) {
    uint32_t row = block_row + threadIdx.y;
    T block_total = init;

    T* row_src = src_ + row * row_size;
    T* row_tgt = tgt_ + row * row_size;

    // Perform scan on one block at a time, keeping track of the total value of
    // all blocks processed so far.
    for (uint32_t block_col = 0; block_col < row_size; block_col += 2 * num_threads_x) {
      // Load data into shared memory (two values per thread).
      uint32_t col1 = block_col + threadIdx.x;
      uint32_t col2 = block_col + num_threads_x + threadIdx.x;
      if (row < num_rows) {
        if (col1 < row_size) {
          row_buf[threadIdx.x] = row_src[col1];
        } else {
          row_buf[threadIdx.x] = init;
        }

        if (col2 < row_size) {
          row_buf[num_threads_x + threadIdx.x] = row_src[col2];
        } else {
          row_buf[num_threads_x + threadIdx.x] = init;
        }

        // Add the total value of all previous blocks to the first value of this block.
        if (threadIdx.x == 0) {
          row_buf[0] = BinaryFunc<T>()(row_buf[0], block_total);
        }
      }
      __syncthreads();

      for (uint32_t s = num_threads_x, d = 1; s >= 1; s >>= 1, d <<= 1) {
        if (row < num_rows && threadIdx.x < s) {
          uint32_t offset = (2 * threadIdx.x + 1) * d - 1;
          row_buf[offset + d] = BinaryFunc<T>()(row_buf[offset], row_buf[offset + d]);
        }
        __syncthreads();
      }

      for (uint32_t s = 2, d = num_threads_x / 2; d >= 1; s <<= 1, d >>= 1) {
        if (row < num_rows && threadIdx.x < s - 1) {
          uint32_t offset = 2 * (threadIdx.x + 1) * d - 1;
          row_buf[offset + d] = BinaryFunc<T>()(row_buf[offset], row_buf[offset + d]);
        }
        __syncthreads();
      }
      // Write back to output.
      if (row < num_rows) {
        if (col1 < row_size) row_tgt[col1] = row_buf[threadIdx.x];
        if (col2 < row_size) row_tgt[col2] = row_buf[num_threads_x + threadIdx.x];
      }
      block_total = row_buf[2 * num_threads_x - 1];
      __syncthreads();
    }
  }
}

template <typename T, int num_threads_x, int num_threads_y, template <typename> class BinaryFunc>
__global__ void ScanInnerMostDimKernel(const T* in_ptr, T* out_ptr, const int64_t num_rows, const int64_t row_size,
                                       T init) {
  __shared__ T sbuf[num_threads_y][2 * num_threads_x];
  T* row_buf = sbuf[threadIdx.y];
  ScanInnerMostDimKernelImpl<T, num_threads_x, num_threads_y, BinaryFunc>(row_buf, const_cast<T*>(in_ptr), out_ptr,
                                                                          num_rows, row_size, init);
}

template <typename T, template <typename> class BinaryFunctor>
void ScanInnerMostDim(const T* in_ptr, T* out_ptr, const int64_t num_rows, const int64_t row_size, cudaDeviceProp* prop,
                      cudaStream_t cuda_stream) {
  dim3 block(16, 32);
  const int64_t max_grid_dim = prop->maxGridSize[0];
  dim3 grid(std::min(max_grid_dim, CeilDiv(num_rows, (int64_t)block.y)));
  if (std::is_same<BinaryFunctor<T>, SumFunctor<T>>::value) {
    ScanInnerMostDimKernel<T, 16, 32, SumFunctor><<<grid, block, 0, cuda_stream>>>(in_ptr, out_ptr, num_rows, row_size,
                                                                                   /*init*/ 0);
  } else if (std::is_same<BinaryFunctor<T>, ProdFunctor<T>>::value) {
    ScanInnerMostDimKernel<T, 16, 32, ProdFunctor><<<grid, block, 0, cuda_stream>>>(in_ptr, out_ptr, num_rows, row_size,
                                                                                    /*init*/ 1);
  } else {
    // UNIMPLEMENTED() << "Only Support cumsum and cumprod for now.";
  }
}

template <typename T, template <typename> class BinaryFunc>
void CubInclusiveScan(void* temp_storage, size_t temp_storage_bytes, const T* in_ptr, T* out_ptr, int64_t elem_cnt,
                      cudaStream_t cuda_stream) {
  cub::DeviceScan::InclusiveScan(temp_storage, temp_storage_bytes, in_ptr, out_ptr, BinaryFunc<T>(), elem_cnt,
                                 cuda_stream);
  temp_storage = get_temp_buffer(temp_storage_bytes);
  cub::DeviceScan::InclusiveScan(temp_storage, temp_storage_bytes, in_ptr, out_ptr, BinaryFunc<T>(), elem_cnt,
                                 cuda_stream);
}
}  // namespace

template <typename T, template <typename> class BinaryFunc>
void cum_op(const T* in_ptr, tops::ShapeView in_shape, void* temp_buffer, size_t temp_storage_bytes, T* out_ptr,
            int dim, cudaDeviceProp* prop, cudaStream_t cuda_stream) {
  // const auto* in = ctx->Tensor4ArgNameAndIndex("x", 0);
  // auto* out = ctx->Tensor4ArgNameAndIndex("y", 0);
  // const ShapeView& in_shape = in->shape_view();
  // const int64_t dim = ctx->Attr<int64_t>("dim");
  const int64_t dim_size = in_shape.At(dim);

  // Judge whether tensor has 0 size dimension first.
  int elem_cnt = in_shape.elem_cnt();
  if (!elem_cnt) {
    return;
  }

  if (elem_cnt == dim_size) {
    CubInclusiveScan<T, BinaryFunc>(temp_buffer, temp_storage_bytes, in_ptr, out_ptr, elem_cnt, cuda_stream);
  } else if (dim == in_shape.ndims() - 1) {
    // Treat all outer dimension as a single dimension.
    const int64_t num_rows = elem_cnt / dim_size;
    ScanInnerMostDim<T, BinaryFunc>(in_ptr, out_ptr, num_rows, dim_size, prop, cuda_stream);
  } else {
    ScanOuterDim<T, BinaryFunc>(prop, cuda_stream, in_shape, dim, in_ptr, out_ptr);
  }
}

}  // namespace oneflow
