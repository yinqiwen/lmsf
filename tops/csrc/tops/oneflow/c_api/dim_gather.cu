#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <stdexcept>

#include "tops/c_api/c_api.h"
#include "tops/common/err_cuda.h"
#include "tops/common/shape.h"
#include "tops/oneflow/kernel/dim_gather_kernel_util.cuh"

namespace {
template <typename IN_T>
void do_cuda_dim_gather(CTensorView in, int dim, CTensorView indices,
                        cudaStream_t stream, CTensorView out) {
  tops::ShapeView in_shape(in);
  tops::ShapeView index_shape(indices);
  // if (input_tensor->shape_view().elem_cnt() == 0) {
  //   return;
  // }
  // const Tensor* index_tensor = ctx->Tensor4ArgNameAndIndex("index", 0);
  // Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("output", 0);
  // const int32_t dim = ctx->Attr<int32_t>("dim");

  const IN_T *input = reinterpret_cast<const IN_T *>(in.ptr);
  const int32_t *index = reinterpret_cast<const int32_t *>(indices.ptr);
  IN_T *output = reinterpret_cast<IN_T *>(out.ptr);

  // const Shape in_shape = ExpandDimIf0D(input_tensor->shape_view());
  const auto ndim = in_shape.NumAxes();
  const auto dim_length = in_shape.At(dim);

  using IDX_T = int32_t;

  oneflow::DimOpIndexNdHelper<IDX_T> input_nd_helper(in_shape.data(), ndim);
  oneflow::DimOpIndexNdHelper<IDX_T> index_nd_helper(index_shape.data(), ndim);
  oneflow::user_op::DimGatherFunctor<IN_T, IDX_T>()(
      stream, input_nd_helper, index_nd_helper, ndim, index_shape.elem_cnt(),
      dim_length, dim, index, input, output);
}

} // namespace

extern "C" {
void cuda_dim_gather_tensor(CTensorView input, int dim, CTensorView index,
                            cudaStream_t stream, CTensorView output) {
  switch (input.dtype) {
  case ScalarType::DATA_F16: {
    do_cuda_dim_gather<half>(input, dim, index, stream, output);
    break;
  }
  case ScalarType::DATA_F32: {
    do_cuda_dim_gather<float>(input, dim, index, stream, output);
    break;
  }
  case ScalarType::DATA_F64: {
    do_cuda_dim_gather<double>(input, dim, index, stream, output);
    break;
  }
  default: {
    throw new std::runtime_error("not supported dtype for cuda_dim_gather");
  }
  }
}
}