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
#pragma once

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "tinfer/log/log.h"

namespace tinfer {

#define CUBLAS_WORKSPACE_SIZE 33554432

/* **************************** type definition ***************************** */

enum CublasDataType {
  FLOAT_DATATYPE = 0,
  HALF_DATATYPE = 1,
  BFLOAT16_DATATYPE = 2,
  INT8_DATATYPE = 3,
  FP8_DATATYPE = 4
};

enum CudaDataType { FP32 = 0, FP16 = 1, BF16 = 2, INT8 = 3, FP8 = 4 };

enum class OperationType { FP32, FP16, BF16, INT8, FP8 };

cudaError_t cuda_get_set_device(int i_device, int* o_device = NULL);

template <typename T>
struct packed_type;
template <>
struct packed_type<float> {
  using type = float;
};  // we don't need to pack float by default
template <>
struct packed_type<half> {
  using type = half2;
};

#ifdef ENABLE_BF16
template <>
struct packed_type<__nv_bfloat16> {
  using type = __nv_bfloat162;
};
#endif

template <typename T>
struct num_elems;
template <>
struct num_elems<float> {
  static constexpr int value = 1;
};
template <>
struct num_elems<float2> {
  static constexpr int value = 2;
};
template <>
struct num_elems<float4> {
  static constexpr int value = 4;
};
template <>
struct num_elems<half> {
  static constexpr int value = 1;
};
template <>
struct num_elems<half2> {
  static constexpr int value = 2;
};
#ifdef ENABLE_BF16
template <>
struct num_elems<__nv_bfloat16> {
  static constexpr int value = 1;
};
template <>
struct num_elems<__nv_bfloat162> {
  static constexpr int value = 2;
};
#endif

template <typename T, int num>
struct packed_as;
template <typename T>
struct packed_as<T, 1> {
  using type = T;
};
template <>
struct packed_as<half, 2> {
  using type = half2;
};
template <>
struct packed_as<float, 2> {
  using type = float2;
};
template <>
struct packed_as<int8_t, 2> {
  using type = int16_t;
};
template <>
struct packed_as<int32_t, 2> {
  using type = int2;
};
template <>
struct packed_as<half2, 1> {
  using type = half;
};
#ifdef ENABLE_BF16
template <>
struct packed_as<__nv_bfloat16, 2> {
  using type = __nv_bfloat162;
};
template <>
struct packed_as<__nv_bfloat162, 1> {
  using type = __nv_bfloat16;
};
#endif

inline __device__ float2 operator*(float2 a, float2 b) { return make_float2(a.x * b.x, a.y * b.y); }
inline __device__ float2 operator*(float2 a, float b) { return make_float2(a.x * b, a.y * b); }

extern bool g_cuda_sync_check_enable;

void cuda_enable_sync_check(bool v);

/* **************************** debug tools ********************************* */
static const char* _cudaGetErrorEnum(cudaError_t error) { return cudaGetErrorString(error); }

static const char* _cudaGetErrorEnum(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "<unknown>";
}

int getSMVersion();

}  // namespace tinfer

#define CUDA_CHECK(val)                                                            \
  do {                                                                             \
    auto result = (val);                                                           \
    if (result) {                                                                  \
      TINFER_CRITICAL("CUDA runtime error:{}", tinfer::_cudaGetErrorEnum(result)); \
    }                                                                              \
  } while (0)

#define CUDA_SYNC_CHECK()                                                            \
  do {                                                                               \
    if (tinfer::g_cuda_sync_check_enable) {                                          \
      cudaDeviceSynchronize();                                                       \
      cudaError_t result = cudaGetLastError();                                       \
      if (result) {                                                                  \
        TINFER_CRITICAL("CUDA runtime error:{}", tinfer::_cudaGetErrorEnum(result)); \
      }                                                                              \
    }                                                                                \
  } while (0)

#define CUDA_OP_SYNC_CHECK(op)                                                                      \
  do {                                                                                              \
    if (tinfer::g_cuda_sync_check_enable) {                                                         \
      cudaDeviceSynchronize();                                                                      \
      cudaError_t result = cudaGetLastError();                                                      \
      if (result) {                                                                                 \
        TINFER_CRITICAL("CUDA runtime error:{} with op:{}", tinfer::_cudaGetErrorEnum(result), op); \
      }                                                                                             \
    }                                                                                               \
  } while (0)
