// #include <torch/extension.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <c10/cuda/CUDAGuard.h>

#include "cuda_compat.h"
#include "cuda_ext.h"
// #include "dispatch_utils.h"
#include "reduction_utils.cuh"
#include <algorithm>

namespace vllm {

// TODO(woosuk): Further optimize this kernel.
template <typename scalar_t>
__global__ void
rms_norm_kernel(scalar_t *__restrict__ out,          // [..., hidden_size]
                const scalar_t *__restrict__ input,  // [..., hidden_size]
                const scalar_t *__restrict__ weight, // [hidden_size]
                const float epsilon, const int num_tokens,
                const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = (float)input[blockIdx.x * hidden_size + idx];
    variance += x * x;
  }
  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)input[blockIdx.x * hidden_size + idx];
    out[blockIdx.x * hidden_size + idx] =
        ((scalar_t)(x * s_variance)) * weight[idx];
  }
}

// TODO: Further optimize this kernel.
template <typename scalar_t>
__global__ void
fused_add_rms_norm_kernel(scalar_t *__restrict__ input,    // [..., hidden_size]
                          scalar_t *__restrict__ residual, // [..., hidden_size]
                          const scalar_t *__restrict__ weight, // [hidden_size]
                          const float epsilon, const int num_tokens,
                          const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)input[blockIdx.x * hidden_size + idx];
    x += (float)residual[blockIdx.x * hidden_size + idx];
    variance += x * x;
    residual[blockIdx.x * hidden_size + idx] = (scalar_t)x;
  }
  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)residual[blockIdx.x * hidden_size + idx];
    input[blockIdx.x * hidden_size + idx] =
        ((scalar_t)(x * s_variance)) * weight[idx];
  }
}

} // namespace vllm

// void rms_norm(torch::Tensor &out,    // [..., hidden_size]
//               torch::Tensor &input,  // [..., hidden_size]
//               torch::Tensor &weight, // [hidden_size]
//               float epsilon) {
//   int hidden_size = input.size(-1);
//   int num_tokens = input.numel() / hidden_size;

//   dim3 grid(num_tokens);
//   dim3 block(std::min(hidden_size, 1024));
//   const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
//   const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
//   VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "rms_norm_kernel", [&] {
//     vllm::rms_norm_kernel<scalar_t><<<grid, block, 0, stream>>>(
//         out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
//         weight.data_ptr<scalar_t>(), epsilon, num_tokens, hidden_size);
//   });
// }

// void fused_add_rms_norm(torch::Tensor &input,    // [..., hidden_size]
//                         torch::Tensor &residual, // [..., hidden_size]
//                         torch::Tensor &weight,   // [hidden_size]
//                         float epsilon) {
//   int hidden_size = input.size(-1);
//   int num_tokens = input.numel() / hidden_size;

//   dim3 grid(num_tokens);
//   dim3 block(std::min(hidden_size, 1024));
//   const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
//   const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
//   VLLM_DISPATCH_FLOATING_TYPES(
//       input.scalar_type(), "fused_add_rms_norm_kernel", [&] {
//         vllm::fused_add_rms_norm_kernel<scalar_t><<<grid, block, 0,
//         stream>>>(
//             input.data_ptr<scalar_t>(), residual.data_ptr<scalar_t>(),
//             weight.data_ptr<scalar_t>(), epsilon, num_tokens, hidden_size);
//       });
// }

extern "C" {
struct RmsNormKernelParams {
  cudaStream_t stream;
  float epsilon;
  int hidden_size;
  int num_tokens;
  ScalarType dtype;
};
void vllm_rms_norm(void *out, const void *input, const void *weight,
                   RmsNormKernelParams params) {
  dim3 grid(params.num_tokens);
  dim3 block(std::min(params.hidden_size, 1024));
  const cudaStream_t stream = params.stream;
  switch (params.dtype) {
  case DATA_F16: {
    vllm::rms_norm_kernel<half><<<grid, block, 0, stream>>>(
        reinterpret_cast<half *>(out), reinterpret_cast<const half *>(input),
        reinterpret_cast<const half *>(weight), params.epsilon,
        params.num_tokens, params.hidden_size);
    break;
  }
  case DATA_BF16: {
    vllm::rms_norm_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16 *>(out),
        reinterpret_cast<const __nv_bfloat16 *>(input),
        reinterpret_cast<const __nv_bfloat16 *>(weight), params.epsilon,
        params.num_tokens, params.hidden_size);

    break;
  }
  case DATA_F32: {
    vllm::rms_norm_kernel<float><<<grid, block, 0, stream>>>(
        reinterpret_cast<float *>(out), reinterpret_cast<const float *>(input),
        reinterpret_cast<const float *>(weight), params.epsilon,
        params.num_tokens, params.hidden_size);
    break;
  }
  default: {
    break;
  }
  }
}

void vllm_fused_add_rms_norm(void *input, void *residual, const void *weight,
                             RmsNormKernelParams params) {
  dim3 grid(params.num_tokens);
  dim3 block(std::min(params.hidden_size, 1024));
  const cudaStream_t stream = params.stream;
  switch (params.dtype) {
  case DATA_F16: {
    vllm::fused_add_rms_norm_kernel<half><<<grid, block, 0, stream>>>(
        reinterpret_cast<half *>(input), reinterpret_cast<half *>(residual),
        reinterpret_cast<const half *>(weight), params.epsilon,
        params.num_tokens, params.hidden_size);
    break;
  }
  case DATA_BF16: {
    vllm::fused_add_rms_norm_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16 *>(input),
        reinterpret_cast<__nv_bfloat16 *>(residual),
        reinterpret_cast<const __nv_bfloat16 *>(weight), params.epsilon,
        params.num_tokens, params.hidden_size);

    break;
  }
  case DATA_F32: {
    vllm::fused_add_rms_norm_kernel<float><<<grid, block, 0, stream>>>(
        reinterpret_cast<float *>(input), reinterpret_cast<float *>(residual),
        reinterpret_cast<const float *>(weight), params.epsilon,
        params.num_tokens, params.hidden_size);
    break;
  }
  default: {
    break;
  }
  }
}
}
