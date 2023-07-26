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

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"

#include "sentencepiece_processor.h"
#include "tinfer/tokenizer/tokenizer.h"

namespace tinfer {
namespace tokenizer {
// SentencePiece tokenizer. Initialized with a model file.
class SentencePieceTokenizer : public Tokenizer {
public:
  // Initialize the SentencePiece tokenizer from model file path.
  explicit SentencePieceTokenizer(const std::string &path_to_model) {
    sp_.Load(path_to_model);
  }

  explicit SentencePieceTokenizer(const char *spmodel_buffer_data,
                                  size_t spmodel_buffer_size) {
    absl::string_view buffer_binary(spmodel_buffer_data, spmodel_buffer_size);
    sp_.LoadFromSerializedProto(buffer_binary);
  }

  // Perform tokenization, return tokenized results.
  TokenizerResult Tokenize(const std::string &input) override {
    TokenizerResult result;
    std::vector<std::string> &subwords = result.subwords;
    sp_.Encode(input, &subwords);
    return result;
  }

  // Find the id of a string token.
  bool LookupId(std::string_view key, int *result) const override {
    *result = sp_.PieceToId(key);
    return true;
  }

  // Find the string token of an id.
  bool LookupWord(int vocab_id, absl::string_view *result) const override {
    *result = sp_.IdToPiece(vocab_id);
    return true;
  }
  void Encode(const std::string &input, std::vector<int> &ids) override {
    sp_.Encode(input, &ids);
  }
  void Decode(const std::vector<int> &ids, std::string &text) override {
    sp_.Decode(ids, &text);
  }

private:
  sentencepiece::SentencePieceProcessor sp_;
};
} // namespace tokenizer
} // namespace tinfer