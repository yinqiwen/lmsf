/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include "tops/arrayfire/common/array.h"
#include "tops/arrayfire/common/dim4.h"
#include "tops/arrayfire/common/types.h"

namespace arrayfire {
namespace cuda {
template <typename T>
void tile(const Array<T> &in, const arrayfire::dim4 &tileDims, const cudaDeviceProp *props, cudaStream_t stream,
          Array<T> &out);
}  // namespace cuda
}  // namespace arrayfire
