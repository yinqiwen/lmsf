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
#include <stdint.h>
#include <map>
#include <string>
#include <vector>
#include "kcfg_json.h"

#include "tinfer/module/params.h"
#include "tinfer/utils/cuda_utils.h"

namespace tinfer {

using IntParams = std::map<std::string, int32_t>;

struct ModelConfig {
  std::string model_name;
  uint32_t head_num = 0;
  uint32_t size_per_head = 0;
  uint32_t inter_size = 0;
  uint32_t max_seq_len = 0;
  uint32_t max_pos_seq_len = 0;
  uint32_t num_layer = 0;
  uint32_t vocab_size = 0;
  uint32_t start_id = 50256;
  uint32_t end_id = 50256;
  std::string weight_data_type = "fp32";
  uint32_t tensor_para_size = 1;
  uint32_t hidden_units = 0;
  std::string tokenizer = "bert";

  KCFG_DEFINE_FIELDS(model_name, head_num, size_per_head, inter_size, max_seq_len, max_pos_seq_len, num_layer,
                     vocab_size, start_id, end_id, weight_data_type, tensor_para_size, hidden_units, tokenizer)
  CudaDataType data_type = CudaDataType::FP32;

  void Init();
};

struct ModelLayerWeightConfig {
  uint32_t count = 0;
  std::map<std::string, std::vector<size_t>> weights;
  KCFG_DEFINE_FIELDS(count, weights)
};
struct ModelWeightConfig {
  std::map<std::string, std::vector<size_t>> common;
  ModelLayerWeightConfig layers;
  KCFG_DEFINE_FIELDS(common, layers)
};

struct ModelOpConfig {
  std::string name;
  Params args;
  bool is_layer_op = false;
  bool once = false;
  KCFG_DEFINE_FIELDS(name, is_layer_op, once)
};

struct ModelDesc {
  std::vector<ModelOpConfig> ops;

  int Load(const std::string& file);
};

}  // namespace tinfer