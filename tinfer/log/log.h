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
#include <memory>
#include <string>

#define SPDLOG_FMT_EXTERNAL 1
#include "spdlog/spdlog.h"

#include "fmt/ostream.h"

namespace tinfer {
extern std::shared_ptr<spdlog::logger> g_default_looger;
void set_default_logger(std::shared_ptr<spdlog::logger> logger);
spdlog::logger *get_default_raw_logger();

}  // namespace tinfer

#define TINFER_DEBUG(...)                                                                                           \
  do {                                                                                                              \
    auto _local_logger_ = tinfer::g_default_looger ? tinfer::g_default_looger.get() : spdlog::default_logger_raw(); \
    if (nullptr != _local_logger_ && _local_logger_->should_log(spdlog::level::debug)) {                            \
      SPDLOG_LOGGER_DEBUG(_local_logger_, __VA_ARGS__);                                                             \
    }                                                                                                               \
  } while (0)

#define TINFER_INFO(...)                                                                                            \
  do {                                                                                                              \
    auto _local_logger_ = tinfer::g_default_looger ? tinfer::g_default_looger.get() : spdlog::default_logger_raw(); \
    if (nullptr != _local_logger_ && _local_logger_->should_log(spdlog::level::info)) {                             \
      SPDLOG_LOGGER_INFO(_local_logger_, __VA_ARGS__);                                                              \
    }                                                                                                               \
  } while (0)

#define TINFER_WARN(...)                                                                                            \
  do {                                                                                                              \
    auto _local_logger_ = tinfer::g_default_looger ? tinfer::g_default_looger.get() : spdlog::default_logger_raw(); \
    if (nullptr != _local_logger_ && _local_logger_->should_log(spdlog::level::warn)) {                             \
      SPDLOG_LOGGER_WARN(_local_logger_, __VA_ARGS__);                                                              \
    }                                                                                                               \
  } while (0)

#define TINFER_ERROR(...)                                                                                           \
  do {                                                                                                              \
    auto _local_logger_ = tinfer::g_default_looger ? tinfer::g_default_looger.get() : spdlog::default_logger_raw(); \
    if (nullptr != _local_logger_ && _local_logger_->should_log(spdlog::level::err)) {                              \
      SPDLOG_LOGGER_ERROR(_local_logger_, __VA_ARGS__);                                                             \
    }                                                                                                               \
  } while (0)

#define TINFER_CRITICAL(...)                                                                                        \
  do {                                                                                                              \
    auto _local_logger_ = tinfer::g_default_looger ? tinfer::g_default_looger.get() : spdlog::default_logger_raw(); \
    if (nullptr != _local_logger_ && _local_logger_->should_log(spdlog::level::critical)) {                         \
      SPDLOG_LOGGER_CRITICAL(_local_logger_, __VA_ARGS__);                                                          \
    }                                                                                                               \
  } while (0)

#define CHECK(val)                        \
  do {                                    \
    bool is_valid_val = (val);            \
    if (!is_valid_val) {                  \
      TINFER_CRITICAL("Assertion fail."); \
    }                                     \
  } while (0)
#define CHECK_WITH_INFO(val, info)                          \
  do {                                                      \
    bool is_valid_val = (val);                              \
    if (!is_valid_val) {                                    \
      TINFER_CRITICAL("Assertion fail with info:{}", info); \
    }                                                       \
  } while (0)
