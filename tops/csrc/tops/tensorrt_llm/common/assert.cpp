/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tops/tensorrt_llm/common/assert.h"

bool CHECK_DEBUG_ENABLED = false;

namespace {

#if !defined(_MSC_VER)
__attribute__((constructor))
#endif
void initOnLoad()
{
  auto constexpr kDebugEnabled = "TRT_LLM_DEBUG_MODE";
  auto const debugEnabled = std::getenv(kDebugEnabled);
  if (debugEnabled && debugEnabled[0] == '1') {
    CHECK_DEBUG_ENABLED = true;
  }
}
}  // namespace
