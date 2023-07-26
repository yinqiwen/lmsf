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
#include "tinfer/tokenizer/bert_tokenizer.h"

#include "tinfer/utils/regex_split.h"

namespace tinfer {
namespace tokenizer {
FlatHashMapBackedWordpiece::FlatHashMapBackedWordpiece(
    const std::vector<std::string> &vocab)
    : vocab_{vocab} {
  for (int i = 0; i < vocab_.size(); ++i) {
    index_map_[vocab_[i]] = i;
  }
}

LookupStatus FlatHashMapBackedWordpiece::Contains(absl::string_view key,
                                                  bool *value) const {
  *value = index_map_.contains(key);
  return LookupStatus();
}

bool FlatHashMapBackedWordpiece::LookupId(const absl::string_view key,
                                          int *result) const {
  auto it = index_map_.find(key);
  if (it == index_map_.end()) {
    return false;
  }
  *result = it->second;
  return true;
}

bool FlatHashMapBackedWordpiece::LookupWord(int vocab_id,
                                            absl::string_view *result) const {
  if (vocab_id >= vocab_.size() || vocab_id < 0) {
    return false;
  }
  *result = vocab_[vocab_id];
  return true;
}
TokenizerResult BertTokenizer::Tokenize(const std::string &input) {
  return TokenizeWordpiece(input);
}

WordpieceTokenizerResult
BertTokenizer::TokenizeWordpiece(const std::string &input) const {
  WordpieceTokenizerResult result;
  std::vector<std::string> &subwords = result.subwords;
  std::vector<int> &wp_absolute_begin_offset = result.wp_begin_offset;
  std::vector<int> &wp_absolute_end_offset = result.wp_end_offset;

  std::vector<absl::string_view> tokens;
  std::vector<long long> begin_offsets;
  std::vector<long long> end_offsets;

  // Run through tokenize function
  ::tinfer::RegexSplit(input, delim_re_, true, include_delim_re_, &tokens,
                       &begin_offsets, &end_offsets);

  for (int token_index = 0; token_index < tokens.size(); token_index++) {
    auto &token = tokens[token_index];

    int num_word_pieces = 0;
    LookupStatus status = WordpieceTokenize(
        token, options_.max_bytes_per_token, options_.max_chars_per_subtoken,
        options_.suffix_indicator, options_.use_unknown_token,
        options_.unknown_token, options_.split_unknown_chars, &vocab_,
        &subwords, &wp_absolute_begin_offset, &wp_absolute_end_offset,
        &num_word_pieces);

    result.row_lengths.emplace_back(num_word_pieces);

    // for the last num_word_pieces added into wp_absolute_begin_offset and
    // wp_absolute_end_offset, offset them with begin_offsets[token_index]
    int absolute_offset_size = wp_absolute_begin_offset.size();
    for (int i = num_word_pieces; i > 0; i--) {
      wp_absolute_begin_offset[absolute_offset_size - i] +=
          begin_offsets[token_index];
      wp_absolute_end_offset[absolute_offset_size - i] +=
          begin_offsets[token_index];
    }
    if (!status.success) {
      return result;
    }
  }

  return result;
}

} // namespace tokenizer
} // namespace tinfer