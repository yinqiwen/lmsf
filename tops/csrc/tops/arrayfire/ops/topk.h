/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include "tops/arrayfire/common/array.h"
#include "tops/arrayfire/common/types.h"

namespace arrayfire {
namespace cuda {
template <typename T>
void topk(Array<T> &keys, Array<uint32_t> &vals, const Array<T> &in,
          const int k, const int dim, const arrayfire::topkFunction order,
          cudaStream_t stream);
} // namespace cuda
} // namespace arrayfire
