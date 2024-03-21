#define _USE_MATH_DEFINES
#include "cuda_utils.cuh"
#include <math.h>
#include <stdint.h>

#define UNARY_OP(TYPENAME, FN_NAME)                                            \
  extern "C" __global__ void FN_NAME(                                          \
      const size_t numel, const size_t num_dims, const size_t *info,           \
      TYPENAME *out, TYPENAME x) {                                             \
    const size_t *dims = info;                                                 \
    const size_t *strides = info + num_dims;                                   \
    if (is_contiguous(num_dims, dims, strides)) {                              \
      for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;  \
           i += blockDim.x * gridDim.x) {                                      \
        out[i] = x;                                                            \
      }                                                                        \
    } else {                                                                   \
      for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;  \
           i += blockDim.x * gridDim.x) {                                      \
        unsigned strided_i = get_strided_index(i, num_dims, dims, strides);    \
        out[strided_i] = x;                                                    \
      }                                                                        \
    }                                                                          \
  }

#define UNARY_OP1(TYPENAME, FN_NAME, FUNC)                                     \
  extern "C" __global__ void FN_NAME(                                          \
      const size_t numel, const size_t num_dims, const size_t *info,           \
      const TYPENAME *inp, TYPENAME *out) {                                    \
    const size_t *dims = info;                                                 \
    const size_t *strides = info + num_dims;                                   \
    if (is_contiguous(num_dims, dims, strides)) {                              \
      for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;  \
           i += blockDim.x * gridDim.x) {                                      \
        TYPENAME x = inp ? inp[i] : out[i];                                    \
        out[i] = FUNC;                                                         \
      }                                                                        \
    } else {                                                                   \
      for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;  \
           i += blockDim.x * gridDim.x) {                                      \
        unsigned strided_i = get_strided_index(i, num_dims, dims, strides);    \
        TYPENAME x = inp ? inp[i] : out[i];                                    \
        out[strided_i] = FUNC;                                                 \
      }                                                                        \
    }                                                                          \
  }

#if __CUDA_ARCH__ >= 800
UNARY_OP(__nv_bfloat16, assign_bf16)
UNARY_OP1(__nv_bfloat16, copy_bf16, x)
#endif

#if __CUDA_ARCH__ >= 530
UNARY_OP(__half, assign_f16)
UNARY_OP1(__nv_bfloat16, copy_f16, x)
#endif

UNARY_OP(uint8_t, assign_u8)
UNARY_OP(uint32_t, assign_u32)
UNARY_OP(int64_t, assign_i64)
UNARY_OP(float, assign_f32)
UNARY_OP(double, assign_f64)
UNARY_OP1(uint8_t, copy_u8, x)
UNARY_OP1(uint32_t, copy_u32, x)
UNARY_OP1(int64_t, copy_i64, x)
UNARY_OP1(float, copy_f32, x)
UNARY_OP1(double, copy_f64, x)
