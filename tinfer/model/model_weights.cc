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
#include "tinfer/model/model_weights.h"
#include <cuda_fp16.h>

#include "absl/strings/str_join.h"
#include "tinfer/log/log.h"
#include "tinfer/utils/mem_utils.h"

namespace tinfer {
const void *ModelWeights::Get(const std::string &name) const {
  auto found = weights_.find(name);
  if (found != weights_.end()) {
    return found->second->weight;
  }
  return nullptr;
}
int ModelWeights::Load(const std::string &dir, const ModelWeightsOptions &options) {
  options_ = options;
  dir_ = dir;
  std::string config_file = dir + "/config.json";
  if (!kcfg::ParseFromJsonFile(config_file, config_)) {
    TINFER_ERROR("Failed to load config file:{}", config_file);
    return -1;
  }
  config_.Init();

  std::string weight_config_file = dir + "/model_weights.json";
  if (!kcfg::ParseFromJsonFile(weight_config_file, weight_config_)) {
    TINFER_ERROR("Failed to load config file:{}", weight_config_file);
    return -1;
  }
  std::map<std::string, std::vector<size_t>> name_shapes = weight_config_.common;

  for (uint32_t i = 0; i < weight_config_.layers.count; i++) {
    for (const auto &[name, shape] : weight_config_.layers.weights) {
      name_shapes["layers." + std::to_string(i) + "." + name] = shape;
    }
  }

  for (const auto &[name, shape] : name_shapes) {
    TINFER_INFO("Load weights:{} with shape:[{}]", name, absl::StrJoin(shape, ","));
    switch (options_.load_data_type) {
      case CudaDataType::FP32: {
        AddModelParameter<float>(name, shape);
        break;
      }
      case CudaDataType::FP16: {
        AddModelParameter<half>(name, shape);
        break;
      }
      default: {
        TINFER_ERROR("Unsupported load data type:{}", options_.load_data_type);
        return -1;
      }
    }
  }
  size_t used, total;
  cuda_get_mem_usage(used, total);
  TINFER_INFO("GPU mem used:{}MB, total:{}MB", used * 1.0 / 1024 / 1024, total * 1.0 / 1024 / 1024);

  return 0;
}
template <typename T>
ModelParameter<T>::ModelParameter(const std::string &w_name, const std::vector<size_t> &w_shape) {
  name = w_name;
  shape = w_shape;
}

template <typename T>
int ModelParameter<T>::Load(const std::string &dir, CudaDataType serving_data_type, CudaDataType file_data_type) {
  size_t weight_len = 1;
  for (auto v : shape) {
    weight_len *= v;
  }
  std::string file_path = dir + "/model." + name + ".bin";
  cuda_malloc(&_internal_weight, weight_len, false);
  load_weight_from_bin<T>(_internal_weight, shape, file_path, file_data_type);
  weight = _internal_weight;
  return 0;
}
template <typename T>
ModelParameter<T>::~ModelParameter() {
  cuda_free(_internal_weight);
}

template struct ModelParameter<float>;
template struct ModelParameter<half>;
#ifdef ENABLE_BF16
template struct ModelParameter<__nv_bfloat16>;
#endif

}  // namespace tinfer