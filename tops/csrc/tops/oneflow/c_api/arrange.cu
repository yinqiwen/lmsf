#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <stdexcept>

#include "tops/c_api/c_api.h"
#include "tops/common/err_cuda.h"
#include "tops/common/shape.h"
#include "tops/oneflow/kernel/arange_kernel_util.cuh"

extern "C" {
void cuda_arrange_int_tensor(int64_t start, int64_t delta,
                             const int64_t arange_elem_cnt, CTensorView out,
                             cudaStream_t stream) {
  switch (out.dtype) {
  case ScalarType::DATA_U8: {
    oneflow::user_op::ArangeFunctor<uint8_t> func;
    func(stream, start, delta, arange_elem_cnt,
         reinterpret_cast<uint8_t *>(out.ptr));
    break;
  }
  case ScalarType::DATA_U32: {
    oneflow::user_op::ArangeFunctor<uint32_t> func;
    func(stream, start, delta, arange_elem_cnt,
         reinterpret_cast<uint32_t *>(out.ptr));
    break;
  }
  case ScalarType::DATA_I64: {
    oneflow::user_op::ArangeFunctor<int64_t> func;
    func(stream, start, delta, arange_elem_cnt,
         reinterpret_cast<int64_t *>(out.ptr));
    break;
  }

  default: {
    throw new std::runtime_error("not supported dtype for cuda_arrange_tensor");
  }
  }
}

void cuda_arrange_float_tensor(double start, double delta,
                               const int64_t arange_elem_cnt, CTensorView out,
                               cudaStream_t stream) {
  switch (out.dtype) {
  case ScalarType::DATA_F16: {
    oneflow::user_op::ArangeFunctor<half> func;
    func(stream, static_cast<half>(start), static_cast<half>(delta),
         arange_elem_cnt, reinterpret_cast<half *>(out.ptr));
    break;
  }
  case ScalarType::DATA_F32: {
    oneflow::user_op::ArangeFunctor<float> func;
    func(stream, static_cast<float>(start), static_cast<float>(delta),
         arange_elem_cnt, reinterpret_cast<float *>(out.ptr));
    break;
  }
  case ScalarType::DATA_F64: {
    oneflow::user_op::ArangeFunctor<double> func;
    func(stream, start, delta, arange_elem_cnt,
         reinterpret_cast<double *>(out.ptr));
    break;
  }

  default: {
    throw new std::runtime_error("not supported dtype for cuda_arrange_tensor");
  }
  }
}
}