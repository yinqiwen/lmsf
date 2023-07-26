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
#include "tinfer/model/model.h"
#include "tinfer/log/log.h"
#include "tinfer/tokenizer/bert_tokenizer.h"
#include "tinfer/tokenizer/sentencepiece_tokenizer.h"

namespace tinfer {
Model::Model(ModelWeights& weights) : weights_(weights) {}
int Model::Init(const ModelDesc& desc) {
  if (weights_.GetConig().tokenizer == "bert") {
    tokenizer_ = std::make_unique<::tinfer::tokenizer::BertTokenizer>(weights_.GetDir() + "/vocab.txt");
  } else if (weights_.GetConig().tokenizer == "sentencepiece") {
    tokenizer_ = std::make_unique<::tinfer::tokenizer::SentencePieceTokenizer>(weights_.GetDir() + "/tokenizer.model");
  } else {
    TINFER_ERROR("Unsupoorted tokenizer:{}!!!", weights_.GetConig().tokenizer);
    return -1;
  }

  for (const auto& op : desc.ops) {
    auto* op_instance = ModuleFactory::GetOperator(op.name, "");
    if (nullptr == op_instance) {
      op_instance = ModuleFactory::GetOperator(op.name, weights_.GetConig().weight_data_type);
    }
    if (nullptr == op_instance) {
      TINFER_ERROR("No operator:{} found!!!", op.name);
      return -1;
    }
    OperationExecUnit unit;
    unit.op = op_instance;
    unit.config = op;
    ops_.emplace_back(unit);
  }
  ctx_.Init(&weights_);
  return 0;
}
void Model::Encode(const std::vector<std::string>& txts, std::vector<int>& start_ids, std::vector<int>& start_lengths,
                   int& max_input_len) {
  std::vector<std::vector<int>> tmp_start_ids;
  for (const auto& txt : txts) {
    std::vector<int> ids;
    tokenizer_->Encode(txt, ids);
    // ids = model_->tokenizer->AddSpecialToken(ids);
    // ids.pop_back();
    tmp_start_ids.push_back(ids);
    start_lengths.push_back(ids.size());
  }

  max_input_len = start_lengths[0];
  for (uint i = 1; i < (uint)start_lengths.size(); i++) {
    max_input_len = max_input_len > start_lengths[i] ? max_input_len : start_lengths[i];
  }

  // Add padding
  for (int i = 0; i < (int)tmp_start_ids.size(); i++) {
    for (int j = (int)tmp_start_ids[i].size(); j < max_input_len; j++) {
      tmp_start_ids[i].push_back(weights_.GetConig().end_id);
    }
  }
  for (auto& ids : tmp_start_ids) {
    start_ids.insert(start_ids.end(), ids.begin(), ids.end());
  }
}

void Model::ExecuteOperator(OperationExecUnit& op) {
  TINFER_INFO("Execute {}", op.op->Name());
  op.op->PreProcess(ctx_, op.config.args);
  op.op->Forward(ctx_);
  CUDA_OP_SYNC_CHECK(op.op->Name());
  op.op->PostProcess(ctx_);
}

void Model::DoGenerate() {
  std::vector<std::string> txts;
  for (auto& task : ctx_.task->batch) {
    txts.emplace_back(task.request.prompt);
  }
  Encode(txts, ctx_.task->start_ids, ctx_.task->start_lengths, ctx_.task->max_input_length);
  ctx_.task->max_request_output_len = 128;
  ctx_.task->batch_size = ctx_.task->batch.size();
  ctx_.task->max_session_len = ctx_.task->max_request_output_len + ctx_.task->max_input_length;

  for (size_t op_idx = 0; op_idx < ops_.size(); op_idx++) {
    auto& op_unit = ops_[op_idx];
    if (op_unit.config.once) {
      if (!op_unit.once) {
        ExecuteOperator(op_unit);
        op_unit.once = true;
      }
      continue;
    }
    if (op_unit.config.is_layer_op) {
      size_t layer_op_idx = op_idx;
      for (size_t layer = 0; layer < weights_.GetConig().num_layer; layer++) {
        layer_op_idx = op_idx;
        do {
          ExecuteOperator(ops_[layer_op_idx]);
          layer_op_idx++;
        } while (layer_op_idx < ops_.size() && ops_[layer_op_idx].config.is_layer_op);
      }
      op_idx = (layer_op_idx - 1);
      continue;
    }
    ExecuteOperator(op_unit);
  }
}

int Model::Generate(const GenerateRequest& req, GenerateCallback&& callback) {
  ctx_.task = std::make_unique<BatchGenerateTask>();
  ctx_.task->batch.emplace_back(GenerateTask{req, std::move(callback)});
  DoGenerate();
  return 0;
}

}  // namespace tinfer