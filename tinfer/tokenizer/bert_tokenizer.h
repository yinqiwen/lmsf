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

#include "absl/container/flat_hash_map.h"
#include "re2/re2.h"

#include "tinfer/tokenizer/tokenizer.h"
#include "tinfer/tokenizer/wordpiece_tokenizer.h"
#include "tinfer/utils/vocab_utils.h"

namespace tinfer {
namespace tokenizer {

constexpr char kDefaultDelimRe[] =
    R"((\s+|[!-/]|[:-@]|[\[-`]|[{-~]|[\p{P}]|[\x{4E00}-\x{9FFF}]|[\x{3400}-\x{4DBF}]|[\x{20000}-\x{2A6DF}]|[\x{2A700}-\x{2B73F}]|[\x{2B740}-\x{2B81F}]|[\x{2B820}-\x{2CEAF}]|[\x{F900}-\x{FAFF}]|[\x{2F800}-\x{2FA1F}]))";
constexpr char kDefaultIncludeDelimRe[] =
    R"(([!-/]|[:-@]|[\[-`]|[{-~]|[\p{P}]|[\x{4E00}-\x{9FFF}]|[\x{3400}-\x{4DBF}]|[\x{20000}-\x{2A6DF}]|[\x{2A700}-\x{2B73F}]|[\x{2B740}-\x{2B81F}]|[\x{2B820}-\x{2CEAF}]|[\x{F900}-\x{FAFF}]|[\x{2F800}-\x{2FA1F}]))";
constexpr int kDefaultMaxBytesPerToken = 100;
constexpr int kDefaultMaxCharsPerSubToken = 100;
constexpr char kDefaultSuffixIndicator[] = "##";
constexpr bool kDefaultUseUnknownToken = true;
constexpr char kDefaultUnknownToken[] = "[UNK]";
constexpr bool kDefaultSplitUnknownChars = false;

// Result of wordpiece tokenization including subwords and offsets.
// Example:
// input:                tokenize     me  please
// subwords:             token ##ize  me  plea ##se
// wp_begin_offset:     [0,      5,   9,  12,    16]
// wp_end_offset:       [     5,    8,  11,   16,  18]
// row_lengths:         [2,          1,  1]
struct WordpieceTokenizerResult : TokenizerResult {
  std::vector<int> wp_begin_offset;
  std::vector<int> wp_end_offset;
  std::vector<int> row_lengths;
};
// Options to create a BertTokenizer.
struct BertTokenizerOptions {
  int max_bytes_per_token = kDefaultMaxBytesPerToken;
  int max_chars_per_subtoken = kDefaultMaxCharsPerSubToken;
  std::string suffix_indicator = kDefaultSuffixIndicator;
  bool use_unknown_token = kDefaultUseUnknownToken;
  std::string unknown_token = kDefaultUnknownToken;
  bool split_unknown_chars = kDefaultSplitUnknownChars;
  std::string delim_str = kDefaultDelimRe;
  std::string include_delim_str = kDefaultIncludeDelimRe;
};

// A flat-hash-map based implementation of WordpieceVocab, used in
// BertTokenizer to invoke tensorflow::text::WordpieceTokenize within.
class FlatHashMapBackedWordpiece : public WordpieceVocab {
public:
  explicit FlatHashMapBackedWordpiece(const std::vector<std::string> &vocab);

  LookupStatus Contains(const std::string_view key, bool *value) const override;
  bool LookupId(absl::string_view key, int *result) const;
  bool LookupWord(int vocab_id, absl::string_view *result) const;
  int VocabularySize() const { return vocab_.size(); }

private:
  // All words indexed position in vocabulary file.
  std::vector<std::string> vocab_;
  absl::flat_hash_map<absl::string_view, int> index_map_;
};

// Wordpiece tokenizer for bert models. Initialized with a vocab file or vector.
class BertTokenizer : public Tokenizer {
public:
  // Initialize the tokenizer from vocab vector and tokenizer configs.
  explicit BertTokenizer(const std::vector<std::string> &vocab,
                         const BertTokenizerOptions &options = {})
      : vocab_{FlatHashMapBackedWordpiece(vocab)}, options_{options},
        delim_re_{options.delim_str}, include_delim_re_{
                                          options.include_delim_str} {}

  // Initialize the tokenizer from file path to vocab and tokenizer configs.
  explicit BertTokenizer(const std::string &path_to_vocab,
                         const BertTokenizerOptions &options = {})
      : BertTokenizer(LoadVocabFromFile(path_to_vocab), options) {}

  // Initialize the tokenizer from buffer and size of vocab and tokenizer
  // configs.
  BertTokenizer(const char *vocab_buffer_data, size_t vocab_buffer_size,
                const BertTokenizerOptions &options = {})
      : BertTokenizer(LoadVocabFromBuffer(vocab_buffer_data, vocab_buffer_size),
                      options) {}

  // Perform tokenization, return tokenized results containing the subwords.
  TokenizerResult Tokenize(const std::string &input) override;

  // Perform tokenization, return wordpiece-specific tokenized result including
  // subwords and offsets
  WordpieceTokenizerResult TokenizeWordpiece(const std::string &input) const;

  // Check if a certain key is included in the vocab.
  LookupStatus Contains(const absl::string_view key, bool *value) const {
    return vocab_.Contains(key, value);
  }

  // Find the id of a wordpiece.
  bool LookupId(std::string_view key, int *result) const override {
    return vocab_.LookupId(key, result);
  }

  // Find the wordpiece from an id.
  bool LookupWord(int vocab_id, absl::string_view *result) const override {
    return vocab_.LookupWord(vocab_id, result);
  }

  int VocabularySize() const { return vocab_.VocabularySize(); }

private:
  FlatHashMapBackedWordpiece vocab_;
  BertTokenizerOptions options_;
  RE2 delim_re_;
  RE2 include_delim_re_;
};

} // namespace tokenizer
} // namespace tinfer