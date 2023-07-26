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
#pragma once

#include <functional>
#include <memory>
#include <vector>
#include "tinfer/model/model_task.h"
#include "tinfer/model/model_weights.h"
#include "tinfer/module/module.h"
#include "tinfer/tokenizer/tokenizer.h"

namespace tinfer {

struct OperationExecUnit {
  Module* op = nullptr;
  ModelOpConfig config;
  bool once = false;
};

class Model {
 public:
  Model(ModelWeights& weights);
  int Init(const ModelDesc& desc);

  int Generate(const GenerateRequest& req, GenerateCallback&& callbak);

 private:
  void ExecuteOperator(OperationExecUnit& op);
  void Encode(const std::vector<std::string>& txts, std::vector<int>& start_ids, std::vector<int>& start_lengths,
              int& max_input_len);

  void DoGenerate();

  ModelWeights& weights_;
  std::vector<OperationExecUnit> ops_;
  Context ctx_;

  std::unique_ptr<::tinfer::tokenizer::Tokenizer> tokenizer_;
};
}  // namespace tinfer