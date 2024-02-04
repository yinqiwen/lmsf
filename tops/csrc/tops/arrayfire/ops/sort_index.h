/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include "tops/arrayfire/common/array.h"

#pragma once

namespace arrayfire {
namespace cuda {
template <typename T>
void sort_index(Array<T> &okey, Array<uint32_t> &oval, const Array<T> &in,
                const uint32_t dim, bool isAscending,
                const cudaDeviceProp *props, cudaStream_t stream);
} // namespace cuda
} // namespace arrayfire
