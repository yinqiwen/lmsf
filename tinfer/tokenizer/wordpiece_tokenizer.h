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
#include <string>
#include <string_view>
#include <vector>

namespace tinfer {
namespace tokenizer {
struct LookupStatus {
  LookupStatus() : error_msg(""), success(true) {}
  LookupStatus(std::string msg) : error_msg(std::move(msg)), success(false) {}
  std::string error_msg;
  bool success;

  static LookupStatus OK() { return LookupStatus(); }
};

class WordpieceVocab {
public:
  virtual ~WordpieceVocab() {}
  virtual LookupStatus Contains(const std::string_view key,
                                bool *value) const = 0;
};

LookupStatus WordpieceTokenize(
    const std::string_view &token, const int max_bytes_per_token,
    const int max_chars_per_subtoken, const std::string &suffix_indicator,
    bool use_unknown_token, const std::string &unknown_token,
    bool split_unknown_characters, const WordpieceVocab *vocab_map,
    std::vector<std::string> *subwords, std::vector<int> *begin_offset,
    std::vector<int> *end_offset, int *num_word_pieces);

// As above but with `max_bytes_per_subtoken` unknown,
// and split_unknown_characters=false. (For backwards compatability.)
LookupStatus WordpieceTokenize(
    const std::string_view &token, const int max_bytes_per_token,
    const std::string &suffix_indicator, bool use_unknown_token,
    const std::string &unknown_token, const WordpieceVocab *vocab_map,
    std::vector<std::string> *subwords, std::vector<int> *begin_offset,
    std::vector<int> *end_offset, int *num_word_pieces);
} // namespace tokenizer
} // namespace tinfer