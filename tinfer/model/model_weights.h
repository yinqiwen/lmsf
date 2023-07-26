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
#pragma once
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"

#include "tinfer/model/model_config.h"
#include "tinfer/utils/cuda_utils.h"

namespace tinfer {

struct BaseModelParameter {
  std::string name;
  std::vector<size_t> shape;
  void *weight = nullptr;

  virtual int Load(const std::string &dir, CudaDataType serving_data_type, CudaDataType file_data_type) = 0;
  virtual ~BaseModelParameter() = default;
};

template <typename T>
struct ModelParameter : public BaseModelParameter {
  T *_internal_weight = nullptr;
  ModelParameter(const std::string &w_name, const std::vector<size_t> &w_shape);

  int Load(const std::string &dir, CudaDataType serving_data_type, CudaDataType file_data_type) override;
  ~ModelParameter();
};

struct ModelWeightsOptions {
  uint32_t max_seq_len = 128;
  CudaDataType load_data_type = CudaDataType::FP32;
};

class ModelWeights {
 public:
  int Load(const std::string &dir, const ModelWeightsOptions &options);
  const ModelConfig &GetConig() const { return config_; }
  const std::string &GetDir() const { return dir_; }
  const void *Get(const std::string &name) const;

 private:
  template <typename T>
  void AddModelParameter(const std::string &name, const std::vector<size_t> &shape) {
    std::unique_ptr<BaseModelParameter> w = std::make_unique<ModelParameter<T>>(name, shape);
    std::string_view w_name = w->name;
    w->Load(dir_, options_.load_data_type, config_.data_type);
    weights_.emplace(w_name, std::move(w));
  }

  std::string dir_;
  ModelConfig config_;
  ModelWeightConfig weight_config_;
  CudaDataType model_file_type_;
  ModelWeightsOptions options_;

  absl::flat_hash_map<std::string_view, std::unique_ptr<BaseModelParameter>> weights_;
};
}  // namespace tinfer