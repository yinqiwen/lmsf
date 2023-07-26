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
#include "tinfer/utils/regex_split.h"

#include <vector>
namespace tinfer {
namespace {

template <typename T>
void RegexSplitImpl(absl::string_view input, const RE2 &re2,
                    bool include_delimiter, const RE2 &include_delim_regex,
                    std::vector<absl::string_view> *tokens,
                    std::vector<T> *begin_offsets,
                    std::vector<T> *end_offsets) {
  absl::string_view leftover = input;
  absl::string_view last_end = leftover;

  // Keep looking for split points until we have reached the end of the input.
  absl::string_view extracted_delim_token;
  while (RE2::FindAndConsume(&leftover, re2, &extracted_delim_token)) {
    absl::string_view token(last_end.data(),
                            extracted_delim_token.data() - last_end.data());
    bool has_non_empty_token = token.length() > 0;
    bool should_include_delim =
        include_delimiter && include_delim_regex.FullMatch(
                                 extracted_delim_token, include_delim_regex);
    last_end = leftover;

    // Mark the end of the previous token, only if there was something.
    if (has_non_empty_token) {
      tokens->push_back(token);
      // Mark the end of the last token
      begin_offsets->push_back(token.data() - input.data());
      end_offsets->push_back(token.data() + token.length() - input.data());
    }

    if (should_include_delim) {
      // If desired, include the deliminator as a token.
      tokens->push_back(extracted_delim_token);
      // Mark the end of the token at the end of the beginning of the delimiter.
      begin_offsets->push_back(extracted_delim_token.data() - input.data());
      end_offsets->push_back(extracted_delim_token.data() +
                             extracted_delim_token.length() - input.data());
    }
  }

  // Close the last token.
  if (!leftover.empty()) {
    tokens->push_back(leftover);
    begin_offsets->push_back(leftover.data() - input.data());
    end_offsets->push_back(leftover.data() + leftover.length() - input.data());
  }
}

} // namespace

void RegexSplit(absl::string_view input, const RE2 &re2, bool include_delimiter,
                const RE2 &include_delim_regex,
                std::vector<absl::string_view> *tokens,
                std::vector<long> *begin_offsets, // NOLINT
                std::vector<long> *end_offsets) { // NOLINT
  RegexSplitImpl(input, re2, include_delimiter, include_delim_regex, tokens,
                 begin_offsets, end_offsets);
}

void RegexSplit(absl::string_view input, const RE2 &re2, bool include_delimiter,
                const RE2 &include_delim_regex,
                std::vector<absl::string_view> *tokens,
                std::vector<long long> *begin_offsets, // NOLINT
                std::vector<long long> *end_offsets) { // NOLINT
  RegexSplitImpl(input, re2, include_delimiter, include_delim_regex, tokens,
                 begin_offsets, end_offsets);
}
} // namespace tinfer