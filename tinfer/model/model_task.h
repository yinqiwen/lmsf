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
#include <string>
#include <vector>

namespace tinfer {
struct GenerateRequest {
  std::string prompt;

  int min_length = 0;
  int beam_width = 1;
  int request_output_len = 128;
  int top_k = 3;
  float top_p = 1.0;
  float beam_search_diversity_rate = 0.0;
  float temperature = 1.0;
  float len_penalty = 0.0;
  float repetition_penalty = 1.0;
  float presence_penalty = 0.0;
  bool stream = false;
};
struct GenerateStat {
  int64_t start_us = 0;
  int64_t batch_queue_wait_us = 0;
  int64_t fetch_inference_instance_us = 0;
  int64_t complete_us = 0;
};
struct GenerateResponse {
  std::vector<std::string> choices;
  GenerateStat stat;
  bool complete = false;
  int err_code = 0;
};

using GenerateCallback = std::function<void(GenerateResponse&&)>;

struct GenerateTask {
  GenerateRequest request;
  GenerateCallback callback;
};

}  // namespace tinfer