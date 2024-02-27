#include <cuda_runtime_api.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include "tops/c_api/c_api.h"
#include "tops/tensorrt_llm/gemm/gemm.h"
// #include "tops/tensorrt_llm/gemm/cublasMMWrapper.h"
// #include "tops/tensorrt_llm/gemm_test/gemm_profiler.h"
using namespace tensorrt_llm::gemm;
using namespace tensorrt_llm::common;
void test_gemm() {
  Gemm* gemm = (Gemm*)new_gemm(0, 0, DATA_F16);
  CShapeView weight;
  weight.ndim = 2;
  weight.shape[0] = 4096;
  weight.shape[1] = 11008;
  CShapeView min_input, max_input;
  min_input.ndim = 3;
  min_input.shape[0] = 1;
  min_input.shape[1] = 1;
  min_input.shape[2] = 11008;
  max_input.ndim = 3;
  max_input.shape[0] = 8;
  max_input.shape[1] = 1;
  max_input.shape[2] = 11008;

  gemm->configGemm(CublasDataType::HALF_DATATYPE, false, true, min_input, max_input, weight);

  int rc = save_gemm_config();
  printf("save rc:%d\n", rc);
}

// void load_gemm() {
//   int rc = load_gemm_config("./gemm_config");
//   printf("load rc:%d\n", rc);
// }

// void test1() {
//   Gemm* gemm = (Gemm*)new_gemm(0, 0, DATA_F16);
//   CShapeView weight;
//   weight.ndim = 2;
//   weight.shape[0] = 12288;
//   weight.shape[1] = 4096;
//   CShapeView min_input, max_input;
//   min_input.ndim = 3;
//   min_input.shape[0] = 1;
//   min_input.shape[1] = 9;
//   min_input.shape[2] = 4096;

//   max_input.ndim = 3;
//   max_input.shape[0] = 8;
//   max_input.shape[1] = 9;
//   max_input.shape[2] = 4096;
//   gemm->configGemm(CublasDataType::HALF_DATATYPE, false, true, min_input, max_input, weight);

//   half* input_dptr = nullptr;
//   cudaMalloc((void**)&input_dptr, sizeof(half) * 4096);
//   half* weight_dptr = nullptr;
//   cudaMalloc((void**)&weight_dptr, sizeof(half) * 4096 * 12288);
//   half* out_dptr = nullptr;
//   cudaMalloc((void**)&out_dptr, sizeof(half) * 12288);

//   CTensorView in;
//   in.dtype = ScalarType::DATA_F16;
//   in.ptr = input_dptr;
//   in.shape[0] = 1;
//   in.shape[1] = 1;
//   in.shape[2] = 4096;
//   in.shape[3] = 1;
//   in.ndim = 3;
//   CTensorView weight_t;
//   weight_t.dtype = ScalarType::DATA_F16;
//   weight_t.ptr = weight_dptr;
//   weight_t.shape[0] = 12288;
//   weight_t.shape[1] = 4096;
//   weight_t.shape[2] = 1;
//   weight_t.shape[3] = 1;
//   weight_t.ndim = 2;
//   CTensorView out;
//   out.dtype = ScalarType::DATA_F16;
//   out.ptr = out_dptr;
//   out.shape[0] = 1;
//   out.shape[1] = 1;
//   out.shape[2] = 12288;
//   out.shape[3] = 1;
//   out.ndim = 3;

//   gemm->gemm(0, 1, in, weight_t, out);
// }

// void test2() {
//   fastertransformer::GemmProfiler<half> profiler;

//   CShapeView weight;
//   weight.ndim = 2;
//   weight.shape[0] = 4096;
//   weight.shape[1] = 11008;
//   CShapeView input;
//   input.ndim = 3;
//   input.shape[0] = 1;
//   input.shape[1] = 1;
//   input.shape[2] = 11008;

//   int nbDimsA = input.ndim;
//   int nbDimsB = weight.ndim;

//   const auto M = fastertransformer::cublasMMWrapper::computeMDimension(0, nbDimsA, input.shape);
//   const auto N = fastertransformer::cublasMMWrapper::computeNDimension(1, nbDimsB, weight.shape);
//   const auto K = 0 ? input.shape[0] : input.shape[nbDimsA - 1];
//   cublasOperation_t transa, transb;
//   int m, n, k;
//   int lda, ldb, ldc;
//   fastertransformer::cublasMMWrapper::getProblemParams(transa, transb, m, n, k, lda, ldb, ldc, 0, 1, M, N, K);
//   profiler.add_gemm("test", m, n, k, 1, lda, ldb, ldc);
//   profiler.generate_gemm_config("gemm_config.in");
// }

int main() {
  test_gemm();
  //  load_gemm();
  // test1();

  return 0;
}