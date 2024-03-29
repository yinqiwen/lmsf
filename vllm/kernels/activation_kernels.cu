// #include <ATen/cuda/CUDAContext.h>
// #include <torch/extension.h>
// #include <c10/cuda/CUDAGuard.h>

#include "cuda_compat.h"
#include "cuda_ext.h"
#include <algorithm>
// #include "dispatch_utils.h"

namespace vllm {

// Activation and gating kernel template.
template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t &)>
__global__ void
act_and_mul_kernel(scalar_t *__restrict__ out,         // [..., d]
                   const scalar_t *__restrict__ input, // [..., 2, d]
                   const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = VLLM_LDG(&input[token_idx * 2 * d + idx]);
    const scalar_t y = VLLM_LDG(&input[token_idx * 2 * d + d + idx]);
    out[token_idx * d + idx] = ACT_FN(x) * y;
  }
}

template <typename T> __device__ __forceinline__ T silu(const T &x) {
  // x * sigmoid(x)
  return (T)(((float)x) / (1.0f + expf((float)-x)));
}

template <typename scalar_t>
__global__ void
silu_and_mul_kernel(scalar_t *__restrict__ out,         // [..., d]
                    const scalar_t *__restrict__ input, // [..., 2, d]
                    const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = VLLM_LDG(&input[token_idx * 2 * d + idx]);
    const scalar_t y = VLLM_LDG(&input[token_idx * 2 * d + d + idx]);
    out[token_idx * d + idx] = silu(x) * y;
  }
}

template <typename T> __device__ __forceinline__ T gelu_kernel(const T &x) {
  // Equivalent to PyTorch GELU with 'none' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L38
  const float f = (float)x;
  constexpr float ALPHA = M_SQRT1_2;
  return (T)(f * 0.5f * (1.0f + ::erf(f * ALPHA)));
}

} // namespace vllm

extern "C" {
struct ActivationKernelParams {
  cudaStream_t stream;
  int d;
  int num_tokens;
  ScalarType dtype;
};
void vllm_silu_and_mul(void *out, const void *input,
                       ActivationKernelParams params) {
  int64_t num_tokens = params.num_tokens;
  int d = params.d;
  const cudaStream_t stream = params.stream;
  dim3 grid(num_tokens);
  dim3 block(std::min(d, 1024));
  switch (params.dtype) {
  case DATA_F16: {
    vllm::silu_and_mul_kernel<half>
        <<<grid, block, 0, stream>>>(reinterpret_cast<half *>(out),
                                     reinterpret_cast<const half *>(input), d);
    break;
  }
  case DATA_BF16: {
    vllm::silu_and_mul_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16 *>(out),
        reinterpret_cast<const __nv_bfloat16 *>(input), d);

    break;
  }
  case DATA_F32: {
    vllm::silu_and_mul_kernel<float>
        <<<grid, block, 0, stream>>>(reinterpret_cast<float *>(out),
                                     reinterpret_cast<const float *>(input), d);
    break;
  }
  default: {
    break;
  }
  }
}

void vllm_gelu_and_mul(void *out, const void *input,
                       ActivationKernelParams params) {
  int64_t num_tokens = params.num_tokens;
  int d = params.d;
  const cudaStream_t stream = params.stream;
  dim3 grid(num_tokens);
  dim3 block(std::min(d, 1024));
  switch (params.dtype) {
  case DATA_F16: {
    vllm::act_and_mul_kernel<half, vllm::gelu_kernel<half>>
        <<<grid, block, 0, stream>>>(reinterpret_cast<half *>(out),
                                     reinterpret_cast<const half *>(input), d);
    break;
  }
  case DATA_BF16: {
    vllm::act_and_mul_kernel<__nv_bfloat16, vllm::gelu_kernel<__nv_bfloat16>>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16 *>(out),
            reinterpret_cast<const __nv_bfloat16 *>(input), d);

    break;
  }
  case DATA_F32: {
    vllm::act_and_mul_kernel<float, vllm::gelu_kernel<float>>
        <<<grid, block, 0, stream>>>(reinterpret_cast<float *>(out),
                                     reinterpret_cast<const float *>(input), d);
    break;
  }
  default: {
    break;
  }
  }
}
}

// void silu_and_mul(torch::Tensor &out,   // [..., d]
//                   torch::Tensor &input) // [..., 2 * d]
// {
//   int64_t num_tokens = input.numel() / input.size(-1);
//   int d = input.size(-1) / 2;

//   dim3 grid(num_tokens);
//   dim3 block(std::min(d, 1024));
//   const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
//   const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
//   VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "silu_and_mul_kernel",
//   [&] {
//     vllm::silu_and_mul_kernel<scalar_t><<<grid, block, 0, stream>>>(
//         out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), d);
//   });
// }

namespace vllm {

// Element-wise activation kernel template.
template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t &)>
__global__ void
activation_kernel(scalar_t *__restrict__ out,         // [..., d]
                  const scalar_t *__restrict__ input, // [..., d]
                  const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = VLLM_LDG(&input[token_idx * d + idx]);
    out[token_idx * d + idx] = ACT_FN(x);
  }
}

} // namespace vllm

// Launch element-wise activation kernel.
#define LAUNCH_ACTIVATION_KERNEL(KERNEL)                                       \
  int d = input.size(-1);                                                      \
  int64_t num_tokens = input.numel() / d;                                      \
  dim3 grid(num_tokens);                                                       \
  dim3 block(std::min(d, 1024));                                               \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));            \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();                \
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "activation_kernel", [&] { \
    vllm::activation_kernel<scalar_t, KERNEL<scalar_t>>                        \
        <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),                 \
                                     input.data_ptr<scalar_t>(), d);           \
  });

namespace vllm {

template <typename T> __device__ __forceinline__ T gelu_new_kernel(const T &x) {
  const float x3 = (float)(x * x * x);
  const T t = (T)tanhf((T)(0.79788456f * (float)(x + (T)(0.044715f * x3))));
  return ((T)0.5) * x * (((T)1.0) + t);
}

template <typename T>
__device__ __forceinline__ T gelu_fast_kernel(const T &x) {
  const float f = (float)x;
  const T t =
      (T)tanhf(((T)(f * 0.79788456f)) * (((T)1.0) + (T)(0.044715f * f) * x));
  return ((T)0.5) * x * (((T)1.0) + t);
}

} // namespace vllm

// void gelu_new(torch::Tensor &out,   // [..., d]
//               torch::Tensor &input) // [..., d]
// {
//   LAUNCH_ACTIVATION_KERNEL(vllm::gelu_new_kernel);
// }

// void gelu_fast(torch::Tensor &out,   // [..., d]
//                torch::Tensor &input) // [..., d]
// {
//   LAUNCH_ACTIVATION_KERNEL(vllm::gelu_fast_kernel);
// }
