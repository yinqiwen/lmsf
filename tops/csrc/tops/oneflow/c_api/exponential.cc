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

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <stdexcept>

#include "tops/c_api/c_api.h"
#include "tops/oneflow/kernel/distribution/exponential_distribution.h"
#include "tops/oneflow/random/cuda_random_generator.h"

extern "C" {

static std::shared_ptr<oneflow::ep::CUDAGenerator> gen;

static std::shared_ptr<oneflow::ep::CUDAGenerator> getGenerator() {
  if (!gen) {
    gen = std::make_shared<oneflow::ep::CUDAGenerator>(0, 0);
  }
  return gen;
}

void cuda_reset_random_seed(uint64_t seed) {
  gen = std::make_shared<oneflow::ep::CUDAGenerator>(seed, 0);
}

void cuda_create_exponential_tensor(float lambd, cudaStream_t stream,
                                    CTensorView output) {
  switch (output.dtype) {
  case ScalarType::DATA_F16: {
    oneflow::ExponentialDistribution<oneflow::DeviceType::kCUDA, half> dis(
        static_cast<half>(lambd));

    dis(stream, get_tensor_element_count(&output),
        reinterpret_cast<half *>(output.ptr), getGenerator());

    break;
  }
  case ScalarType::DATA_F32: {
    oneflow::ExponentialDistribution<oneflow::DeviceType::kCUDA, float> dis(
        static_cast<half>(lambd));
    dis(stream, get_tensor_element_count(&output),
        reinterpret_cast<float *>(output.ptr), getGenerator());
    break;
  }
  case ScalarType::DATA_F64: {
    oneflow::ExponentialDistribution<oneflow::DeviceType::kCUDA, double> dis(
        static_cast<half>(lambd));
    dis(stream, get_tensor_element_count(&output),
        reinterpret_cast<double *>(output.ptr), getGenerator());

    break;
  }
  default: {
    throw new std::runtime_error(
        "not supported dtype for cuda_create_exponential_tensor");
  }
  }
}
}