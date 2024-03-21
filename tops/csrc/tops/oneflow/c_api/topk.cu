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
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <stdexcept>

#include "tops/c_api/c_api.h"
#include "tops/common/err_cuda.h"
#include "tops/common/shape.h"
#include "tops/oneflow/common/cuda_util.h"
#include "tops/oneflow/kernel/top_k_kernel.cuh"

namespace {

template <typename T> class TmpBufferManager final {
public:
  // OF_DISALLOW_COPY_AND_MOVE(TmpBufferManager);
  TmpBufferManager(int64_t capacity, void *ptr, const tops::ShapeView &in_shape)
      : capacity_{capacity}, sorted_in_elem_cnt_{in_shape.elem_cnt()},
        indices_elem_cnt_{sorted_in_elem_cnt_}, sorted_indices_elem_cnt_{
                                                    sorted_in_elem_cnt_} {
    const int64_t sorted_in_aligned_bytes =
        oneflow::GetCudaAlignedSize(sorted_in_elem_cnt_ * sizeof(T));
    const int64_t indices_aligned_bytes =
        oneflow::GetCudaAlignedSize(indices_elem_cnt_ * sizeof(int64_t));
    const int64_t sorted_indices_aligned_bytes = indices_aligned_bytes;
    sorted_in_ptr_ = reinterpret_cast<T *>(ptr);
    indices_ptr_ = reinterpret_cast<uint32_t *>(
        reinterpret_cast<char *>(sorted_in_ptr_) + sorted_in_aligned_bytes);
    sorted_indices_ptr_ = reinterpret_cast<uint32_t *>(
        reinterpret_cast<char *>(indices_ptr_) + indices_aligned_bytes);
    temp_storage_ptr_ =
        reinterpret_cast<void *>(reinterpret_cast<char *>(sorted_indices_ptr_) +
                                 sorted_indices_aligned_bytes);
    temp_storage_bytes_ = capacity_ - sorted_in_aligned_bytes -
                          indices_aligned_bytes - sorted_indices_aligned_bytes;
    // CHECK_GE(temp_storage_bytes_, 0);
  }
  ~TmpBufferManager() = default;

  T *SortedInPtr() const { return sorted_in_ptr_; }
  uint32_t *IndicesPtr() const { return indices_ptr_; }
  uint32_t *SortedIndicesPtr() const { return sorted_indices_ptr_; }
  void *TempStoragePtr() const { return temp_storage_ptr_; }

  int64_t TempStorageBytes() const { return temp_storage_bytes_; }

private:
  int64_t capacity_;

  T *sorted_in_ptr_;
  uint32_t *indices_ptr_;
  uint32_t *sorted_indices_ptr_;
  void *temp_storage_ptr_;

  int64_t sorted_in_elem_cnt_;
  int64_t indices_elem_cnt_;
  int64_t sorted_indices_elem_cnt_;
  int64_t temp_storage_bytes_;
};

__global__ void InitializeIndices(int64_t elem_cnt, uint32_t *indices_ptr,
                                  int64_t instance_size) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) { indices_ptr[i] = i % instance_size; };
}

template <typename T>
inline size_t get_temp_storage_bytes(const tops::ShapeView &in_shape,
                                     bool descend) {
  const int64_t elem_cnt = in_shape.elem_cnt();
  const int64_t instance_size = in_shape.At(in_shape.NumAxes() - 1);
  const int64_t instance_num = elem_cnt / instance_size;

  /* Sorted In*/
  const int64_t sorted_in_aligned_bytes =
      oneflow::GetCudaAlignedSize(elem_cnt * sizeof(T));
  /* Indices */
  const int64_t indices_aligned_bytes =
      oneflow::GetCudaAlignedSize(elem_cnt * sizeof(int64_t));
  /* Sorted Indices */
  const int64_t sorted_indices_aligned_bytes = indices_aligned_bytes;
  /* CUB Temp Storage */
  int64_t temp_storage_bytes =
      descend ? oneflow::InferTempStorageForSortPairsDescending<T, int64_t>(
                    instance_num, instance_size)
              : oneflow::InferTempStorageForSortPairsAscending<T, int64_t>(
                    instance_num, instance_size);

  return sorted_in_aligned_bytes + indices_aligned_bytes +
         sorted_indices_aligned_bytes + temp_storage_bytes;
}

template <typename T>
void do_cuda_topk_indices(CTensorView input, int k, bool descend,
                          cudaStream_t stream, CTensorView output) {
  tops::ShapeView in_shape(input);
  if (in_shape.elem_cnt() == 0) {
    return;
  }

  const int64_t elem_cnt = in_shape.elem_cnt();
  const int64_t instance_size = in_shape.At(in_shape.NumAxes() - 1);
  const int64_t instance_num = elem_cnt / instance_size;
  k = std::min(static_cast<int64_t>(k), instance_size);

  if (k > 30 || instance_num == 1) {
    size_t tmp_capacity = get_temp_storage_bytes<T>(in_shape, descend);
    void *tmp_buffer = get_temp_buffer(tmp_capacity);
    TmpBufferManager<T> buf_manager(static_cast<int32_t>(tmp_capacity),
                                    tmp_buffer, in_shape);

    InitializeIndices<<<oneflow::BlocksNum4ThreadsNum(elem_cnt),
                        oneflow::kCudaThreadsNumPerBlock, 0, stream>>>(
        elem_cnt, buf_manager.IndicesPtr(), instance_size);
    if (descend) {
      oneflow::SortPairsDescending(
          reinterpret_cast<const T *>(input.ptr), buf_manager.IndicesPtr(),
          instance_num, instance_size, buf_manager.TempStoragePtr(),
          buf_manager.TempStorageBytes(), buf_manager.SortedInPtr(),
          buf_manager.SortedIndicesPtr(), stream);
    } else {
      oneflow::SortPairsAscending(
          reinterpret_cast<const T *>(input.ptr), buf_manager.IndicesPtr(),
          instance_num, instance_size, buf_manager.TempStoragePtr(),
          buf_manager.TempStorageBytes(), buf_manager.SortedInPtr(),
          buf_manager.SortedIndicesPtr(), stream);
    }
    OF_CUDA_CHECK(cudaMemcpy2DAsync(
        reinterpret_cast<uint32_t *>(output.ptr), k * sizeof(uint32_t),
        buf_manager.SortedIndicesPtr(), instance_size * sizeof(uint32_t),
        k * sizeof(uint32_t), instance_num, cudaMemcpyDefault, stream));
  } else {
    // Use as many heaps as possible (# of heaps == # of threads used in thread
    // block). Limitation 1: size of shared memory We also need heap_size *
    // num_heap to be pow-of-2 which is necessary for bitonic sort
    const int64_t heap_size = oneflow::topk::PowOf2Ceil(k, 16);
    int32_t num_heap = oneflow::topk::PowOf2Floor(
        oneflow::kCudaMaxSharedMemoryByteSize /
            (heap_size * sizeof(oneflow::topk::Entry<T>)),
        16);
    // Limitation 2: # of threads in thread block
    num_heap = std::min(num_heap, oneflow::kCudaThreadsNumPerBlock);

    if (descend) {
      oneflow::topk::HeapTopKKernel<T>
          <<<instance_num, num_heap,
             num_heap * heap_size * sizeof(oneflow::topk::Entry<T>), stream>>>(
              reinterpret_cast<const T *>(input.ptr), instance_num,
              instance_size, k, heap_size, oneflow::GetMaxVal<int64_t>(),
              oneflow::GetMinVal<T>(),
              reinterpret_cast<uint32_t *>(output.ptr));
    } else {
      oneflow::topk::HeapMinTopKKernel<T>
          <<<instance_num, num_heap,
             num_heap * heap_size * sizeof(oneflow::topk::Entry<T>), stream>>>(
              reinterpret_cast<const T *>(input.ptr), instance_num,
              instance_size, k, heap_size, oneflow::GetMaxVal<int64_t>(),
              oneflow::GetMinVal<T>(),
              reinterpret_cast<uint32_t *>(output.ptr));
    }
  }
}

} // namespace

extern "C" {
using namespace oneflow;
void cuda_topk_indices(CTensorView input, int k, bool descend,
                       cudaStream_t stream, CTensorView indices) {
  switch (input.dtype) {
  case ScalarType::DATA_U8: {
    do_cuda_topk_indices<uint8_t>(input, k, descend, stream, indices);
    break;
  }
  case ScalarType::DATA_U32: {
    do_cuda_topk_indices<uint32_t>(input, k, descend, stream, indices);
    break;
  }
  case ScalarType::DATA_I64: {
    do_cuda_topk_indices<int64_t>(input, k, descend, stream, indices);
    break;
  }
  case ScalarType::DATA_F16: {
    do_cuda_topk_indices<half>(input, k, descend, stream, indices);
    break;
  }
  case ScalarType::DATA_F32: {
    do_cuda_topk_indices<float>(input, k, descend, stream, indices);
    break;
  }
  case ScalarType::DATA_F64: {
    do_cuda_topk_indices<double>(input, k, descend, stream, indices);
    break;
  }
  default: {
    throw new std::runtime_error("not supported dtype for cuda_topk_indices");
  }
  }
}
}