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
#include "tinfer/utils/vocab_utils.h"

#include <fstream>

#include "absl/strings/str_split.h"

namespace tinfer {
struct membuf : std::streambuf {
  membuf(char *begin, char *end) { this->setg(begin, begin, end); }
};

void ReadIStreamLineByLine(
    std::istream *istream,
    const std::function<void(std::string)> &line_processor) {
  std::string str;
  while (std::getline(*istream, str)) {
    if (!str.empty()) {
      if (str.back() == '\r') { // Remove \r on Windows
        line_processor(str.substr(0, str.length() - 1));
      } else {
        line_processor(str);
      }
    }
  }
}

absl::node_hash_map<std::string, int>
ReadIStreamLineSplits(std::istream *istream) {
  absl::node_hash_map<std::string, int> vocab_index_map;
  std::string str;
  ReadIStreamLineByLine(istream, [&vocab_index_map](const std::string &str) {
    std::vector<std::string> v = absl::StrSplit(str, ' ');
    vocab_index_map[v[0]] = std::stoi(v[1]);
  });
  return vocab_index_map;
}

std::vector<std::string> ReadIStreamByLine(std::istream *istream) {
  std::vector<std::string> vocab_from_file;
  std::string str;

  ReadIStreamLineByLine(istream, [&vocab_from_file](const std::string &str) {
    vocab_from_file.push_back(str);
  });
  return vocab_from_file;
}

std::vector<std::string> LoadVocabFromFile(const std::string &path_to_vocab) {
  std::vector<std::string> vocab_from_file;
  // std::string file_name = *PathToResourceAsFile(path_to_vocab);
  std::string file_name = path_to_vocab;
  std::ifstream in(file_name.c_str());
  return ReadIStreamByLine(&in);
}

std::vector<std::string> LoadVocabFromBuffer(const char *vocab_buffer_data,
                                             const size_t vocab_buffer_size) {
  membuf sbuf(const_cast<char *>(vocab_buffer_data),
              const_cast<char *>(vocab_buffer_data + vocab_buffer_size));
  std::istream in(&sbuf);
  return ReadIStreamByLine(&in);
}

absl::node_hash_map<std::string, int>
LoadVocabAndIndexFromFile(const std::string &path_to_vocab) {
  absl::node_hash_map<std::string, int> vocab_index_map;
  //   std::string file_name = *PathToResourceAsFile(path_to_vocab);
  std::string file_name = path_to_vocab;
  std::ifstream in(file_name.c_str());
  return ReadIStreamLineSplits(&in);
}

absl::node_hash_map<std::string, int>
LoadVocabAndIndexFromBuffer(const char *vocab_buffer_data,
                            const size_t vocab_buffer_size) {
  membuf sbuf(const_cast<char *>(vocab_buffer_data),
              const_cast<char *>(vocab_buffer_data + vocab_buffer_size));
  absl::node_hash_map<std::string, int> vocab_index_map;
  std::istream in(&sbuf);
  return ReadIStreamLineSplits(&in);
}

} // namespace tinfer