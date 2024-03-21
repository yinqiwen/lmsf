#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <stdexcept>

#include "tops/c_api/c_api.h"
#include "tops/common/err_cuda.h"
#include "tops/common/shape.h"
#include "tops/oneflow/kernel/where_kernel_util.cuh"

namespace {
template <typename T>
void do_cuda_masked_fill(CTensorView input, CTensorView mask, T scalar_operand,
                         cudaStream_t stream, CTensorView out) {
  tops::ShapeView out_shape(out);
  oneflow::WhereKernelUtil<T, uint8_t>::WhereXScalar(
      stream, out_shape.elem_cnt(), reinterpret_cast<const uint8_t *>(mask.ptr),
      scalar_operand, reinterpret_cast<const T *>(input.ptr),
      reinterpret_cast<T *>(out.ptr));
}
} // namespace

extern "C" {
void cuda_masked_fill_int(CTensorView input, CTensorView mask,
                          int64_t scalar_operand, cudaStream_t stream,
                          CTensorView out) {
  switch (out.dtype) {
  case ScalarType::DATA_U8: {
    do_cuda_masked_fill<uint8_t>(
        input, mask, static_cast<uint8_t>(scalar_operand), stream, out);
    break;
  }
  case ScalarType::DATA_U32: {
    do_cuda_masked_fill<uint32_t>(
        input, mask, static_cast<uint32_t>(scalar_operand), stream, out);
    break;
  }
  case ScalarType::DATA_I64: {
    do_cuda_masked_fill<int64_t>(input, mask, scalar_operand, stream, out);
    break;
  }

  default: {
    throw new std::runtime_error(
        "not supported dtype for cuda_masked_fill_int");
  }
  }
}
void cuda_masked_fill_int_(CTensorView input, CTensorView mask,
                           int64_t scalar_operand, cudaStream_t stream) {
  cuda_masked_fill_int(input, mask, scalar_operand, stream, input);
}

void cuda_masked_fill_float(CTensorView input, CTensorView mask,
                            double scalar_operand, cudaStream_t stream,
                            CTensorView out) {
  switch (out.dtype) {
  case ScalarType::DATA_F16: {
    do_cuda_masked_fill<half>(input, mask, static_cast<half>(scalar_operand),
                              stream, out);
    break;
  }
  case ScalarType::DATA_F32: {
    do_cuda_masked_fill<float>(input, mask, static_cast<float>(scalar_operand),
                               stream, out);
    break;
  }
  case ScalarType::DATA_F64: {
    do_cuda_masked_fill<double>(input, mask, scalar_operand, stream, out);
    break;
  }

  default: {
    throw new std::runtime_error(
        "not supported dtype for cuda_masked_fill_float");
  }
  }
}
void cuda_masked_fill_float_(CTensorView input, CTensorView mask,
                             double scalar_operand, cudaStream_t stream) {
  cuda_masked_fill_float(input, mask, scalar_operand, stream, input);
}
}