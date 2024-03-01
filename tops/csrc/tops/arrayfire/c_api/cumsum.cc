#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include "tops/arrayfire/common/array.h"
#include "tops/arrayfire/ops/scan.h"
#include "tops/c_api/c_api.h"
extern "C" {
// void cuda_cumsum_tensor(CTensorView input, uint32_t dim, cudaStream_t stream, CTensorView output) {
//   switch (input.dtype) {
//     case ScalarType::DATA_F16: {
//       auto input_array = arrayfire::cuda::Array<__half>::create(input);
//       auto output_array = arrayfire::cuda::Array<__half>::create(output);
//       arrayfire::cuda::scan<af_add_t, __half, __half>(output_array, input_array, dim, true, getCudaDeviceProp(),
//                                                       stream);
//       break;
//     }
//     case ScalarType::DATA_F32: {
//       auto input_array = arrayfire::cuda::Array<float>::create(input);
//       auto output_array = arrayfire::cuda::Array<float>::create(output);
//       arrayfire::cuda::scan<af_add_t, float, float>(output_array, input_array, dim, true, getCudaDeviceProp(),
//       stream); break;
//     }
//     case ScalarType::DATA_F64: {
//       auto input_array = arrayfire::cuda::Array<double>::create(input);
//       auto output_array = arrayfire::cuda::Array<double>::create(output);
//       arrayfire::cuda::scan<af_add_t, double, double>(output_array, input_array, dim, true, getCudaDeviceProp(),
//                                                       stream);
//       break;
//     }
//     default: {
//       throw new std::runtime_error("not supported dtype for cuda_cumsum_tensor");
//     }
//   }
// }
}