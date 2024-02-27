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
#include "tops/tensorrt_llm/gemm/gemm.h"
namespace tensorrt_llm {
namespace gemm {

void Gemm::configGemm(CublasDataType dtype, bool transA, bool transB,
                      CShapeView min_input, CShapeView max_input,
                      CShapeView weight) {
  const int nbDimsA = min_input.ndim;
  const int nbDimsB = weight.ndim;

  const auto minM = CublasLtGemmPluginProfiler::computeMDimension(
      transA, nbDimsA, min_input.shape);
  const auto maxM = CublasLtGemmPluginProfiler::computeMDimension(
      transA, nbDimsA, max_input.shape);
  const auto N = CublasLtGemmPluginProfiler::computeNDimension(transB, nbDimsB,
                                                               weight.shape);
  const auto K = transA ? max_input.shape[0] : max_input.shape[nbDimsA - 1];
  GemmDims mDims = GemmDims{minM, maxM, N, static_cast<int32_t>(K)};
  GemmIdCublas mGemmId = GemmIdCublas(mDims.n, mDims.k, dtype, transA, transB);
  profiler->profileTactics(cublas, dtype, mDims, mGemmId);
}
int Gemm::gemm(int transA, int transB, CTensorView input, CTensorView weight,
               CTensorView output) {
  //
  const int nbDimsA = input.ndim;
  const int nbDimsB = weight.ndim;
  const auto M = CublasLtGemmPluginProfiler::computeMDimension(transA, nbDimsA,
                                                               input.shape);
  const auto N = CublasLtGemmPluginProfiler::computeNDimension(transB, nbDimsB,
                                                               weight.shape);
  const int K = transA ? input.shape[0] : input.shape[nbDimsA - 1];

  GemmIdCublas mGemmId{};
  mGemmId.transA = transA;
  mGemmId.transB = transB;
  mGemmId.n = N;
  mGemmId.k = K;
  mGemmId.dtype = CublasDataType::FLOAT_DATATYPE;
  if (input.dtype == ScalarType::DATA_F16) {
    mGemmId.dtype = CublasDataType::HALF_DATATYPE;
  } else if (input.dtype == ScalarType::DATA_BF16) {
    mGemmId.dtype = CublasDataType::BFLOAT16_DATATYPE;
  }

  auto bestTactic = profiler->getBestConfig(M, mGemmId);
  int rc = bestTactic ? 0 : -1;
  CublasLtGemmPluginProfiler::runGemm(
      M, N, K, transA, transB, cublas, input.ptr, weight.ptr, output.ptr,
      bestTactic, workspace, cublas->getStream());
  return rc;
}
} // namespace gemm
} // namespace tensorrt_llm