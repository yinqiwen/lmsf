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
#include "tinfer/utils/mem_utils.h"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <fstream>

#include "tinfer/log/log.h"
#include "tinfer/utils/cuda_type_utils.cuh"

namespace tinfer {

template <typename T>
void cuda_malloc(T **ptr, size_t size, bool is_random_initialize) {
  CHECK_WITH_INFO(size >= ((size_t)0), "Ask cuda_malloc size " + std::to_string(size) + "< 0 is invalid.");
  CUDA_CHECK(cudaMalloc((void **)(ptr), sizeof(T) * size));
  if (is_random_initialize) {
    cuda_random_uniform(*ptr, size);
  }
}

template void cuda_malloc(float **ptr, size_t size, bool is_random_initialize);
template void cuda_malloc(half **ptr, size_t size, bool is_random_initialize);
#ifdef ENABLE_BF16
template void cuda_malloc(__nv_bfloat16 **ptr, size_t size, bool is_random_initialize);
#endif
template void cuda_malloc(uint16_t **ptr, size_t size, bool is_random_initialize);
template void cuda_malloc(int **ptr, size_t size, bool is_random_initialize);
template void cuda_malloc(bool **ptr, size_t size, bool is_random_initialize);
template void cuda_malloc(char **ptr, size_t size, bool is_random_initialize);
template void cuda_malloc(int8_t **ptr, size_t size, bool is_random_initialize);
#ifdef ENABLE_FP8
template void cuda_malloc(__nv_fp8_e4m3 **ptr, size_t size, bool is_random_initialize);
#endif

template <typename T>
void cuda_memset_zero(T *ptr, size_t size) {
  CUDA_CHECK(cudaMemset(static_cast<void *>(ptr), 0, sizeof(T) * size));
}

template void cuda_memset_zero(float *ptr, size_t size);
template void cuda_memset_zero(half *ptr, size_t size);
template void cuda_memset_zero(int *ptr, size_t size);
template void cuda_memset_zero(uint32_t *ptr, size_t size);
template void cuda_memset_zero(bool *ptr, size_t size);
#ifdef ENABLE_FP8
template void cuda_memset_zero(__nv_fp8_e4m3 *ptr, size_t size);
#endif
#ifdef ENABLE_BF16
template void cuda_memset_zero(__nv_bfloat16 *ptr, size_t size);
#endif

template <typename T>
void cuda_free(T *&ptr) {
  if (ptr != NULL) {
    CUDA_CHECK(cudaFree(ptr));
    ptr = NULL;
  }
}

template void cuda_free(float *&ptr);
template void cuda_free(half *&ptr);
#ifdef ENABLE_BF16
template void cuda_free(__nv_bfloat16 *&ptr);
#endif
template void cuda_free(unsigned short *&ptr);
template void cuda_free(int *&ptr);
template void cuda_free(bool *&ptr);
template void cuda_free(char *&ptr);
template void cuda_free(int8_t *&ptr);
#ifdef ENABLE_FP8
template void cuda_free(__nv_fp8_e4m3 *&ptr);
#endif

template <typename T>
void cuda_fill(T *devptr, size_t size, T value, cudaStream_t stream) {
  T *arr = new T[size];
  std::fill(arr, arr + size, value);
  CUDA_CHECK(cudaMemcpyAsync(devptr, arr, sizeof(T) * size, cudaMemcpyHostToDevice, stream));
  delete[] arr;
}

template void cuda_fill(float *devptr, size_t size, float value, cudaStream_t stream);
template void cuda_fill(half *devptr, size_t size, half value, cudaStream_t stream);
#ifdef ENABLE_BF16
template void cuda_fill(__nv_bfloat16 *devptr, size_t size, __nv_bfloat16 value, cudaStream_t stream);
#endif
template void cuda_fill(int *devptr, size_t size, int value, cudaStream_t stream);
template void cuda_fill(bool *devptr, size_t size, bool value, cudaStream_t stream);

template <typename T>
__global__ void cuda_random_uniform_kernel(T *buffer, const size_t size, const int seq_offset) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandState_t local_state;
  curand_init((unsigned long long int)1337, idx + seq_offset, 0, &local_state);
  for (size_t index = idx; index < size; index += blockDim.x * gridDim.x) {
    buffer[index] = (T)(curand_uniform(&local_state) * 0.2f - 0.1f);
  }
}

template <typename T>
void cuda_random_uniform(T *buffer, const size_t size) {
  static int seq_offset = 0;
  cuda_random_uniform_kernel<T><<<256, 256>>>(buffer, size, seq_offset);
  seq_offset += 256 * 256;
}

template void cuda_random_uniform(float *buffer, const size_t size);
template void cuda_random_uniform(half *buffer, const size_t size);
#ifdef ENABLE_BF16
template void cuda_random_uniform(__nv_bfloat16 *buffer, const size_t size);
#endif
template void cuda_random_uniform(int *buffer, const size_t size);
template void cuda_random_uniform(bool *buffer, const size_t size);
template void cuda_random_uniform(char *buffer, const size_t size);
#ifdef ENABLE_FP8
template void cuda_random_uniform(__nv_fp8_e4m3 *buffer, const size_t size);
#endif

template <typename T>
void cuda_cpy_d2h(T *tgt, const T *src, const size_t size) {
  CUDA_CHECK(cudaMemcpy(tgt, src, sizeof(T) * size, cudaMemcpyDeviceToHost));
}

template void cuda_cpy_d2h(float *tgt, const float *src, size_t size);
template void cuda_cpy_d2h(half *tgt, const half *src, size_t size);
#ifdef ENABLE_BF16
template void cuda_cpy_d2h(__nv_bfloat16 *tgt, const __nv_bfloat16 *src, size_t size);
#endif
template void cuda_cpy_d2h(int *tgt, const int *src, size_t size);
template void cuda_cpy_d2h(bool *tgt, const bool *src, size_t size);
#ifdef ENABLE_FP8
template void cuda_cpy_d2h(__nv_fp8_e4m3 *tgt, const __nv_fp8_e4m3 *src, size_t size);
#endif
template void cuda_cpy_d2h(unsigned long long *tgt, const unsigned long long *src, size_t size);
template void cuda_cpy_d2h(unsigned int *tgt, const unsigned int *src, size_t size);
template void cuda_cpy_d2h(int8_t *tgt, const int8_t *src, size_t size);

template <typename T>
void cuda_cpy_h2d(T *tgt, const T *src, const size_t size) {
  CUDA_CHECK(cudaMemcpy(tgt, src, sizeof(T) * size, cudaMemcpyHostToDevice));
}

template void cuda_cpy_h2d(float *tgt, const float *src, size_t size);
template void cuda_cpy_h2d(half *tgt, const half *src, size_t size);
#ifdef ENABLE_BF16
template void cuda_cpy_h2d(__nv_bfloat16 *tgt, const __nv_bfloat16 *src, size_t size);
#endif
template void cuda_cpy_h2d(int *tgt, const int *src, size_t size);
template void cuda_cpy_h2d(bool *tgt, const bool *src, size_t size);
#ifdef ENABLE_FP8
template void cuda_cpy_h2d(__nv_fp8_e4m3 *tgt, const __nv_fp8_e4m3 *src, size_t size);
#endif
template void cuda_cpy_h2d(unsigned long long *tgt, const unsigned long long *src, size_t size);
template void cuda_cpy_h2d(unsigned int *tgt, const unsigned int *src, size_t size);
template void cuda_cpy_h2d(int8_t *tgt, const int8_t *src, size_t size);

template <typename T>
void cuda_cpy_d2d(T *tgt, const T *src, const size_t size) {
  CUDA_CHECK(cudaMemcpy(tgt, src, sizeof(T) * size, cudaMemcpyDeviceToDevice));
}

template void cuda_cpy_d2d(float *tgt, const float *src, size_t size);
template void cuda_cpy_d2d(half *tgt, const half *src, size_t size);
#ifdef ENABLE_BF16
template void cuda_cpy_d2d(__nv_bfloat16 *tgt, const __nv_bfloat16 *src, size_t size);
#endif
template void cuda_cpy_d2d(int *tgt, const int *src, size_t size);
template void cuda_cpy_d2d(bool *tgt, const bool *src, size_t size);
template void cuda_cpy_d2d(int8_t *tgt, const int8_t *src, size_t size);
#ifdef ENABLE_FP8
template void cuda_cpy_d2d(__nv_fp8_e4m3 *tgt, const __nv_fp8_e4m3 *src, size_t size);
#endif
template void cuda_cpy_d2d(unsigned long long *tgt, const unsigned long long *src, size_t size);

template <typename T_IN, typename T_OUT>
__global__ void cudaD2DcpyConvert(T_OUT *dst, const T_IN *src, const size_t size) {
  for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < size; tid += blockDim.x * gridDim.x) {
    dst[tid] = cuda_cast<T_OUT>(src[tid]);
  }
}

template <typename T_IN, typename T_OUT>
void cuda_cpy_d2d_convert(T_OUT *tgt, const T_IN *src, const size_t size, cudaStream_t stream) {
  cudaD2DcpyConvert<<<256, 256, 0, stream>>>(tgt, src, size);
}

template void cuda_cpy_d2d_convert(int8_t *tgt, const float *src, const size_t size, cudaStream_t stream);
template void cuda_cpy_d2d_convert(float *tgt, const int8_t *src, const size_t size, cudaStream_t stream);
template void cuda_cpy_d2d_convert(float *tgt, const int *src, const size_t size, cudaStream_t stream);
template void cuda_cpy_d2d_convert(half *tgt, const int *src, const size_t size, cudaStream_t stream);
template void cuda_cpy_d2d_convert(float *tgt, const float *src, const size_t size, cudaStream_t stream);
template void cuda_cpy_d2d_convert(half *tgt, const float *src, const size_t size, cudaStream_t stream);
template void cuda_cpy_d2d_convert(float *tgt, const half *src, const size_t size, cudaStream_t stream);
template void cuda_cpy_d2d_convert(uint *tgt, const int *src, const size_t size, cudaStream_t stream);
template void cuda_cpy_d2d_convert(int *tgt, const uint *src, const size_t size, cudaStream_t stream);
template void cuda_cpy_d2d_convert(int *tgt, const float *src, const size_t size, cudaStream_t stream);
template void cuda_cpy_d2d_convert(int *tgt, const half *src, const size_t size, cudaStream_t stream);

#ifdef ENABLE_BF16
template void cuda_cpy_d2d_convert(__nv_bfloat16 *tgt, const float *src, const size_t size, cudaStream_t stream);
template void cuda_cpy_d2d_convert(__nv_bfloat16 *tgt, const int *src, const size_t size, cudaStream_t stream);
template void cuda_cpy_d2d_convert(float *tgt, const __nv_bfloat16 *src, const size_t size, cudaStream_t stream);
template void cuda_cpy_d2d_convert(int *tgt, const __nv_bfloat16 *src, const size_t size, cudaStream_t stream);
#endif  // ENABLE_BF16

// loads data from binary file. If it succeeds, returns a non-empty vector. If
// loading fails or the product of the elements in shape is 0, this function
// will return an empty vector.
template <typename T>
std::vector<T> load_weight_from_bin_helper(std::vector<size_t> shape, std::string filename) {
  if (shape.size() > 2) {
    TINFER_ERROR("shape should have less than two dims.");
    return std::vector<T>();
  }
  size_t dim0 = shape[0], dim1 = 1;
  if (shape.size() == 2) {
    dim1 = shape[1];
  }
  size_t size = dim0 * dim1;
  if (size == 0) {
    TINFER_ERROR("shape is zero, skip loading weight from file {}", filename);
    return std::vector<T>();
  }

  std::vector<T> host_array(size);
  std::ifstream in(filename, std::ios::in | std::ios::binary);
  if (!in.is_open()) {
    TINFER_ERROR("file {} cannot be opened, loading model fails!", filename);
    return std::vector<T>();
  }

  size_t loaded_data_size = sizeof(T) * size;
  in.seekg(0, in.end);
  in.seekg(0, in.beg);

  TINFER_DEBUG("Read {} bytes from {}.", loaded_data_size, filename);
  in.read((char *)host_array.data(), loaded_data_size);

  size_t in_get_size = in.gcount();
  if (in_get_size != loaded_data_size) {
    TINFER_ERROR("file {} only has {}, but request {}, loading model fails!", filename, in_get_size, loaded_data_size);
    return std::vector<T>();
  }
  in.close();
  // If we succeed, return an array with values.
  return host_array;
}

template <typename T, typename T_IN>
int load_weight_from_bin_impl(T *ptr, std::vector<size_t> shape, std::string filename) {
  std::vector<T_IN> host_array = load_weight_from_bin_helper<T_IN>(shape, filename);

  if (host_array.empty()) {
    return 0;
  }

  if (std::is_same<T, T_IN>::value == true) {
    cuda_cpy_h2d(ptr, (T *)host_array.data(), host_array.size());
  } else {
    T_IN *ptr_2 = nullptr;
    cuda_malloc(&ptr_2, host_array.size(), false);
    cuda_cpy_h2d(ptr_2, host_array.data(), host_array.size());
    cuda_cpy_d2d_convert(ptr, ptr_2, host_array.size());
    cuda_free(ptr_2);
  }
  return 0;
}

template <typename T>
int load_weight_from_bin(T *ptr, std::vector<size_t> shape, std::string filename, CudaDataType model_file_type) {
  switch (model_file_type) {
    case CudaDataType::FP32:
      load_weight_from_bin_impl<T, float>(ptr, shape, filename);
      break;
    case CudaDataType::FP16:
      load_weight_from_bin_impl<T, half>(ptr, shape, filename);
      break;
    case CudaDataType::INT8:
      load_weight_from_bin_impl<T, int8_t>(ptr, shape, filename);
      break;
#ifdef ENABLE_BF16
    case FtCudaDataType::BF16:
      load_weight_from_bin_impl<T, __nv_bfloat16>(ptr, shape, filename);
      break;
#endif
#ifdef ENABLE_FP8
    case FtCudaDataType::FP8:
      load_weight_from_bin_impl<T, float>(ptr, shape, filename);
      break;
#endif
    default:
      TINFER_ERROR("Does not support FtCudaDataType={}", model_file_type);
      CHECK(false);
  }
  return 0;
}

template <>
int load_weight_from_bin(int *ptr, std::vector<size_t> shape, std::string filename, CudaDataType model_file_type) {
  load_weight_from_bin_impl<int, int>(ptr, shape, filename);
  return 0;
}

template int load_weight_from_bin(float *ptr, std::vector<size_t> shape, std::string filename,
                                  CudaDataType model_file_type);
template int load_weight_from_bin(half *ptr, std::vector<size_t> shape, std::string filename,
                                  CudaDataType model_file_type);
template int load_weight_from_bin(int8_t *ptr, std::vector<size_t> shape, std::string filename,
                                  CudaDataType model_file_type);
#ifdef ENABLE_BF16
template int load_weight_from_bin(__nv_bfloat16 *ptr, std::vector<size_t> shape, std::string filename,
                                  CudaDataType model_file_type);
#endif
#ifdef ENABLE_FP8
template int load_weight_from_bin(__nv_fp8_e4m3 *ptr, std::vector<size_t> shape, std::string filename,
                                  CudaDataType model_file_type);
#endif
template int load_weight_from_bin(int *ptr, std::vector<size_t> shape, std::string filename,
                                  CudaDataType model_file_type);

void cuda_get_mem_usage(size_t &used_bytes, size_t &total_bytes) {
  size_t free_bytes;

  CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  used_bytes = total_bytes - free_bytes;
  // float free = static_cast<float>(free_bytes) / 1024.0 / 1024.0 / 1024.0;
  // float total = static_cast<float>(total_bytes) / 1024.0 / 1024.0 / 1024.0;
  // float used = total - free;
  // printf("%-20s: free: %5.2f GB, total: %5.2f GB, used: %5.2f GB\n", time.c_str(), free, total, used);
}

}  // namespace tinfer