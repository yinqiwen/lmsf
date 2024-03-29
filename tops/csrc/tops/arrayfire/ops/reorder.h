/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// #include <Array.hpp>
#include "tops/arrayfire/common/array.h"

namespace arrayfire {
namespace cuda {
template <typename T>
Array<T> reorder(const Array<T> &in, const arrayfire::dim4 &rdims,
                 const cudaDeviceProp *props, cudaStream_t stream);
} // namespace cuda
} // namespace arrayfire
