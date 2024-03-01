#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "tops/c_api/c_api.h"
#include "tops/common/err_cuda.h"
#include "tops/common/shape.h"
#include "tops/oneflow/kernel/radix_sort.cuh"

namespace {
template <typename T>
inline size_t get_temp_storage_bytes(const tops::ShapeView& in_shape, bool ascend) {
  const int32_t elem_cnt = in_shape.elem_cnt();
  const int32_t instance_size = in_shape.At(in_shape.ndims() - 1);
  const int32_t instance_num = elem_cnt / instance_size;

  /* Sorted In */
  const int32_t sorted_in_aligned_bytes = oneflow::GetCudaAlignedSize(elem_cnt * sizeof(T));
  /* Indices */
  const int32_t indices_aligned_bytes = oneflow::GetCudaAlignedSize(elem_cnt * sizeof(int32_t));
  /* CUB Temp Storage */
  int32_t temp_storage_bytes = -1;
  if (ascend) {
    temp_storage_bytes = oneflow::InferTempStorageForSortPairsAscending<T, int32_t>(instance_num, instance_size);
  } else {
    temp_storage_bytes = oneflow::InferTempStorageForSortPairsDescending<T, int32_t>(instance_num, instance_size);
  }

  return sorted_in_aligned_bytes + indices_aligned_bytes + temp_storage_bytes;
}

template <typename T>
class TmpBufferManager final {
 public:
  TmpBufferManager(int32_t capacity, void* ptr, const tops::ShapeView& in_shape)
      : capacity_{capacity}, sorted_in_elem_cnt_{in_shape.elem_cnt()}, indices_elem_cnt_{sorted_in_elem_cnt_} {
    const int32_t sorted_in_aligned_bytes = oneflow::GetCudaAlignedSize(sorted_in_elem_cnt_ * sizeof(T));
    const int32_t indices_aligned_bytes = oneflow::GetCudaAlignedSize(indices_elem_cnt_ * sizeof(int32_t));
    sorted_in_ptr_ = reinterpret_cast<T*>(ptr);
    indices_ptr_ = reinterpret_cast<int32_t*>(reinterpret_cast<char*>(sorted_in_ptr_) + sorted_in_aligned_bytes);
    temp_storage_ptr_ = reinterpret_cast<void*>(reinterpret_cast<char*>(indices_ptr_) + indices_aligned_bytes);
    temp_storage_bytes_ = capacity_ - sorted_in_aligned_bytes - indices_aligned_bytes;
    // CHECK_GE(temp_storage_bytes_, 0);
  }
  ~TmpBufferManager() = default;

  T* SortedInPtr() const { return sorted_in_ptr_; }
  int32_t* IndicesPtr() const { return indices_ptr_; }
  void* TempStoragePtr() const { return temp_storage_ptr_; }

  int32_t TempStorageBytes() const { return temp_storage_bytes_; }

 private:
  int32_t capacity_;

  T* sorted_in_ptr_;
  int32_t* indices_ptr_;
  void* temp_storage_ptr_;

  int64_t sorted_in_elem_cnt_;
  int64_t indices_elem_cnt_;
  int32_t temp_storage_bytes_;
};

__global__ void InitializeIndices(int32_t elem_cnt, int32_t* indices_ptr, int32_t instance_size) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) { indices_ptr[i] = i % instance_size; };
}

template <typename T>
void do_cuda_argsort_tensor(CTensorView input, bool ascend, CTensorView indices, cudaStream_t stream) {
  tops::ShapeView in_shape(input);

  size_t tmp_capacity = get_temp_storage_bytes<T>(in_shape, ascend);
  void* tmp_buffer = get_temp_buffer(tmp_capacity);
  TmpBufferManager<T> buf_manager(static_cast<int32_t>(tmp_capacity), tmp_buffer, in_shape);
  const int32_t elem_cnt = in_shape.elem_cnt();
  const int32_t instance_size = in_shape.At(in_shape.ndims() - 1);
  const int32_t instance_num = elem_cnt / instance_size;

  InitializeIndices<<<oneflow::BlocksNum4ThreadsNum(elem_cnt), oneflow::kCudaThreadsNumPerBlock, 0, stream>>>(
      elem_cnt, buf_manager.IndicesPtr(), instance_size);

  if (ascend) {
    oneflow::SortPairsAscending(reinterpret_cast<const T*>(input.ptr), buf_manager.IndicesPtr(), instance_num,
                                instance_size, buf_manager.TempStoragePtr(), buf_manager.TempStorageBytes(),
                                buf_manager.SortedInPtr(), reinterpret_cast<int32_t*>(indices.ptr), stream);
  } else {
    oneflow::SortPairsDescending(reinterpret_cast<const T*>(input.ptr), buf_manager.IndicesPtr(), instance_num,
                                 instance_size, buf_manager.TempStoragePtr(), buf_manager.TempStorageBytes(),
                                 buf_manager.SortedInPtr(), reinterpret_cast<int32_t*>(indices.ptr), stream);
  }
}

}  // namespace

extern "C" {
using namespace oneflow;

void cuda_argsort_tensor(CTensorView input, bool ascend, CTensorView indices, cudaStream_t stream) {
  switch (input.dtype) {
    case ScalarType::DATA_F16: {
      do_cuda_argsort_tensor<half>(input, ascend, indices, stream);
      break;
    }
    case ScalarType::DATA_F32: {
      do_cuda_argsort_tensor<float>(input, ascend, indices, stream);
      break;
    }
    case ScalarType::DATA_F64: {
      do_cuda_argsort_tensor<double>(input, ascend, indices, stream);
      break;
    }
    default: {
      throw new std::runtime_error("not supported dtype for cuda_argsort_tensor");
    }
  }
}
}