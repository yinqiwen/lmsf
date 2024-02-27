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
  void *workspace = nullptr;
  ~CudaHandle() {
    if (nullptr != workspace) {
      cudaFree(workspace);
    }
  }
};

static std::shared_ptr<tensorrt_llm::gemm::CublasLtGemmPluginProfiler> GetCublasLtProfiler() {
  static std::shared_ptr<tensorrt_llm::gemm::CublasLtGemmPluginProfiler> instance;
  if (!instance) {
    instance = std::make_shared<tensorrt_llm::gemm::CublasLtGemmPluginProfiler>();
    instance->load(GEMM_CONFIG);
  }
  return instance;
}
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

// int load_gemm_config(const char *file) {
//   auto profiler = GetCublasLtProfiler();
//   return profiler->load(file);
// }
int save_gemm_config() {
  auto profiler = GetCublasLtProfiler();
  return profiler->save(GEMM_CONFIG);
}

void delete_gemm(void *gemm_) {
  tensorrt_llm::gemm::Gemm *gemm = reinterpret_cast<tensorrt_llm::gemm::Gemm *>(gemm_);
  delete gemm;
}

void *new_gemm(int device, cudaStream_t stream, int dtype) {
  cudaSetDevice(device);
  auto handle = GetCudaHandle(device);
  auto wrapper = std::make_shared<tensorrt_llm::common::CublasMMWrapper>(handle->cublasHandle, handle->cublasLtHandle,
                                                                         nullptr, nullptr);
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
  auto *gemm = new tensorrt_llm::gemm::Gemm(wrapper, GetCublasLtProfiler(), handle->workspace);
  return gemm;
}
void gemm_config(void *gemm_, int dtype, bool transA, bool transB, CShapeView min_input, CShapeView max_input,
                 CShapeView weight) {
  tensorrt_llm::gemm::Gemm *gemm = reinterpret_cast<tensorrt_llm::gemm::Gemm *>(gemm_);
  tensorrt_llm::common::CublasDataType cublas_dtype = tensorrt_llm::common::CublasDataType::FLOAT_DATATYPE;
  switch (dtype) {
    case ScalarType::DATA_F16: {
      cublas_dtype = tensorrt_llm::common::CublasDataType::HALF_DATATYPE;
      break;
    }
    case ScalarType::DATA_BF16: {
      cublas_dtype = tensorrt_llm::common::CublasDataType::BFLOAT16_DATATYPE;
      break;
    }
    case ScalarType::DATA_F32: {
      cublas_dtype = tensorrt_llm::common::CublasDataType::FLOAT_DATATYPE;
      break;
    }
    default: {
      throw new std::runtime_error("unsupported dtype to create gemm_config");
    }
  }
  gemm->configGemm(cublas_dtype, transA, transB, min_input, max_input, weight);
}
int gemm_execute(void *gemm_, int transa, int transb, CTensorView input, CTensorView weight, CTensorView output) {
  tensorrt_llm::gemm::Gemm *gemm = reinterpret_cast<tensorrt_llm::gemm::Gemm *>(gemm_);
  return gemm->gemm(transa, transb, input, weight, output);
}
}