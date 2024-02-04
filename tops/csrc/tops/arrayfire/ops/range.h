/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include "tops/arrayfire/common/array.h"

namespace arrayfire {
namespace cuda {
template <typename T>
void range(const cudaDeviceProp *props, cudaStream_t stream,
           const arrayfire::dim4 &dim, Array<T> &out, const int seq_dim = -1);
} // namespace cuda
} // namespace arrayfire
