#include "cuda_utils.cuh"

#define BINARY_OP_OUT(LTYPENAME, RTYPENAME, OUT_TYPENAME, FN_NAME, FUNC)       \
  extern "C" __global__ void FN_NAME(                                          \
      const size_t numel, const size_t num_dims,                               \
      const size_t *dims_and_strides, const LTYPENAME *lhs,                    \
      const RTYPENAME *rhs, OUT_TYPENAME *out) {                               \
    const size_t *dims = dims_and_strides;                                     \
    const size_t *lhs_strides = dims_and_strides + 1 * num_dims;               \
    const size_t *rhs_strides = dims_and_strides + 2 * num_dims;               \
    bool lhs_cont = is_contiguous(num_dims, dims, lhs_strides);                \
    bool rhs_cont = is_contiguous(num_dims, dims, rhs_strides);                \
    if (lhs_cont && rhs_cont) {                                                \
      for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;  \
           i += blockDim.x * gridDim.x) {                                      \
        OUT_TYPENAME x = lhs[i];                                               \
        OUT_TYPENAME y = rhs[i];                                               \
        out[i] = FUNC;                                                         \
      }                                                                        \
    } else if (lhs_cont) {                                                     \
      for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;  \
           i += blockDim.x * gridDim.x) {                                      \
        unsigned int tmp_i = i;                                                \
        unsigned int rhs_i = 0;                                                \
        for (int d = num_dims - 1; d >= 0; d--) {                              \
          unsigned int i_dim = tmp_i % dims[d];                                \
          rhs_i += i_dim * rhs_strides[d];                                     \
          tmp_i /= dims[d];                                                    \
        }                                                                      \
        OUT_TYPENAME x = lhs[i];                                               \
        OUT_TYPENAME y = rhs[rhs_i];                                           \
        out[i] = FUNC;                                                         \
      }                                                                        \
    } else if (rhs_cont) {                                                     \
      for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;  \
           i += blockDim.x * gridDim.x) {                                      \
        unsigned int tmp_i = i;                                                \
        unsigned int lhs_i = 0;                                                \
        for (int d = num_dims - 1; d >= 0; d--) {                              \
          unsigned int i_dim = tmp_i % dims[d];                                \
          lhs_i += i_dim * lhs_strides[d];                                     \
          tmp_i /= dims[d];                                                    \
        }                                                                      \
        OUT_TYPENAME x = lhs[lhs_i];                                           \
        OUT_TYPENAME y = rhs[i];                                               \
        out[i] = FUNC;                                                         \
      }                                                                        \
    } else {                                                                   \
      for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;  \
           i += blockDim.x * gridDim.x) {                                      \
        unsigned int tmp_i = i;                                                \
        unsigned int lhs_i = 0;                                                \
        unsigned int rhs_i = 0;                                                \
        for (int d = num_dims - 1; d >= 0; d--) {                              \
          unsigned int i_dim = tmp_i % dims[d];                                \
          lhs_i += i_dim * lhs_strides[d];                                     \
          rhs_i += i_dim * rhs_strides[d];                                     \
          tmp_i /= dims[d];                                                    \
        }                                                                      \
        OUT_TYPENAME x = lhs[lhs_i];                                           \
        OUT_TYPENAME y = rhs[rhs_i];                                           \
        out[i] = FUNC;                                                         \
      }                                                                        \
    }                                                                          \
  }

// #define BINARY_OP(LTYPENAME, RTYPENAME, FN_NAME, FUNC)                         \
//   BINARY_OP_OUT(LTYPENAME, RTYPENAME, RTYPENAME, FN_NAME, FUNC)
