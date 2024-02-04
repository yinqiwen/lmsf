/*
** BSD 3-Clause License
**
** Copyright (c) 2023, qiyingwang <qiyingwang@tencent.com>, the respective contributors, as shown by the AUTHORS file.
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
** IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
** DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
** FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
** DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
** SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
** CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
** OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include "tops/c_api/c_api.h"
#include "tops/tensorrt_llm/common/cublasMMWrapper.h"

struct CudaHandle {
  std::shared_ptr<cublasHandle_t> cublasHandle;
  std::shared_ptr<cublasLtHandle_t> cublasLtHandle;
  void* workspace = nullptr;
  ~CudaHandle() {
    if (nullptr != workspace) {
      cudaFree(workspace);
    }
  }
};
// not thread safe
static std::shared_ptr<CudaHandle> GetCudaHandle(int device) {
  static std::unordered_map<int, std::shared_ptr<CudaHandle>> handlers;
  auto found = handlers.find(device);
  if (found != handlers.end()) {
    return found->second;
  }
  auto h = std::make_shared<CudaHandle>();
  auto cublasHandle = std::make_shared<cublasHandle_t>();
  cublasCreate(cublasHandle.get());
  auto cublasLtHandle = std::make_shared<cublasLtHandle_t>();
  cublasLtCreate(cublasLtHandle.get());
  h->cublasHandle = cublasHandle;
  h->cublasLtHandle = cublasLtHandle;
  cudaMalloc(&h->workspace, CUBLAS_WORKSPACE_SIZE);
  handlers[device] = h;
  return h;
};

extern "C" {
void delete_cublas_wrapper(void* cublas_wrapper) {
  tensorrt_llm::common::CublasMMWrapper* cublas =
      reinterpret_cast<tensorrt_llm::common::CublasMMWrapper*>(cublas_wrapper);
  delete cublas;
}
void* new_cublas_wrapper(int device, cudaStream_t stream, int dtype) {
  cudaSetDevice(device);
  auto handle = GetCudaHandle(device);
  // auto cublasHandle = std::make_shared<cublasHandle_t>();
  // cublasCreate(cublasHandle.get());
  // auto cublasLtHandle = std::make_shared<cublasLtHandle_t>();
  // cublasLtCreate(cublasLtHandle.get());
  auto* wrapper =
      new tensorrt_llm::common::CublasMMWrapper(handle->cublasHandle, handle->cublasLtHandle, nullptr, nullptr);
  wrapper->setWorkspace(handle->workspace);
  wrapper->setStream(stream);
  switch (dtype) {
    case ScalarType::DATA_F16: {
      wrapper->setFP16GemmConfig(CUDA_R_16F);
      break;
    }
    case ScalarType::DATA_BF16: {
      wrapper->setBF16GemmConfig(CUDA_R_16BF);
      break;
    }
    case ScalarType::DATA_F32: {
      wrapper->setFP32GemmConfig();
      break;
    }
    default: {
      throw new std::runtime_error("unsupported dtype to create cublas_wrapper");
    }
  }
  return wrapper;
}

static void getProblemParams(cublasOperation_t& transa, cublasOperation_t& transb, int& m, int& n, int& k, int& lda,
                             int& ldb, int& ldc, int transA, int transB, int M, int N, int K) {
  transa = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
  transb = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  m = N;
  n = M;
  k = K;
  lda = transB ? K : N;
  ldb = transA ? M : K;
  ldc = N;
}

static int32_t computeMDimension(int transA, const int32_t nbDims, const int64_t* dims) {
  int32_t M = 1;
  if (transA) {
    for (int i = nbDims - 1; i > 0; --i) {
      M *= dims[i];
    }
  } else {
    for (int i = 0; i < nbDims - 1; ++i) {
      M *= dims[i];
    }
  }
  return M;
}

static int32_t computeNDimension(int transB, const int32_t nbDims, const int64_t* dims) {
  int32_t N = 1;
  if (transB) {
    for (int i = 0; i < nbDims - 1; ++i) {
      N *= dims[i];
    }
  } else {
    for (int i = nbDims - 1; i > 0; --i) {
      N *= dims[i];
    }
  }
  return N;
}

void cublas_gemm(void* cublas_wrapper, int transA, int transB, CTensorView input, CTensorView weight,
                 CTensorView output) {
  tensorrt_llm::common::CublasMMWrapper* cublas =
      reinterpret_cast<tensorrt_llm::common::CublasMMWrapper*>(cublas_wrapper);
  const int nbDimsA = input.ndim;
  const int nbDimsB = weight.ndim;
  const auto M = computeMDimension(transA, nbDimsA, input.shape);
  const auto N = computeNDimension(transB, nbDimsB, weight.shape);
  const int K = transA ? input.shape[0] : input.shape[nbDimsA - 1];

  auto A = weight.ptr;
  auto B = input.ptr;
  auto C = output.ptr;
  cublasOperation_t transa, transb;
  int m, n, k;
  int lda, ldb, ldc;
  getProblemParams(transa, transb, m, n, k, lda, ldb, ldc, transA, transB, M, N, K);
  cublas->createDescriptors(static_cast<cublasOperation_t>(transa), static_cast<cublasOperation_t>(transb), m, n, k,
                            lda, ldb, ldc);
  cublas->Gemm(transa, transb, m, n, k, A, lda, B, ldb, C, ldc);
  cublas->destroyDescriptors();
}
}