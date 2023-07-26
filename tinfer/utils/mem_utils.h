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
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <string>
#include <vector>

#include "tinfer/utils/cuda_utils.h"

namespace tinfer {

template <typename T>
void cuda_malloc(T **ptr, size_t size, bool is_random_initialize = true);

template <typename T>
void cuda_memset_zero(T *ptr, size_t size);

template <typename T>
void cuda_free(T *&ptr);

template <typename T>
void cuda_fill(T *devptr, size_t size, T value, cudaStream_t stream = 0);

template <typename T>
void cuda_random_uniform(T *buffer, const size_t size);

template <typename T>
void cuda_cpy_d2h(T *tgt, const T *src, const size_t size);

template <typename T>
void cuda_cpy_h2d(T *tgt, const T *src, const size_t size);

template <typename T>
void cuda_cpy_d2d(T *tgt, const T *src, const size_t size);

template <typename T_IN, typename T_OUT>
void cuda_cpy_d2d_convert(T_OUT *tgt, const T_IN *src, const size_t size, cudaStream_t stream = 0);

template <typename T>
int load_weight_from_bin(T *ptr, std::vector<size_t> shape, std::string filename,
                         CudaDataType model_file_type = CudaDataType::FP32);

void cuda_get_mem_usage(size_t &used_bytes, size_t &total_bytes);

}  // namespace tinfer