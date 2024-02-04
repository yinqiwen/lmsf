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
#include <cstdint>
#include <cuComplex.h>
#include <cuda_runtime.h>

#define __DH__ __device__ __host__

#define divup(a, b) (((a) + (b) -1) / (b))

namespace arrayfire
{
using dim_t = int64_t;

using cdouble = cuDoubleComplex;
using cfloat = cuFloatComplex;

typedef enum
{
    AF_TOPK_MIN = 1,                                   ///< Top k min values
    AF_TOPK_MAX = 2,                                   ///< Top k max values
    AF_TOPK_STABLE = 4,                                ///< Preserve order of indices for equal values
    AF_TOPK_STABLE_MIN = AF_TOPK_STABLE | AF_TOPK_MIN, ///< Top k min with stable indices
    AF_TOPK_STABLE_MAX = AF_TOPK_STABLE | AF_TOPK_MAX, ///< Top k max with stable indices
    AF_TOPK_DEFAULT = 0                                ///< Default option (max)
} topkFunction;

} // namespace arrayfire