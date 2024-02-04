/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include "tops/arrayfire/common/array.h"
#include "tops/arrayfire/common/optypes.h"

namespace arrayfire {
namespace cuda {
template <af_op_t op, typename Ti, typename To>
void scan(Array<To> &out, const Array<Ti> &in, const int dim,
          bool inclusive_scan, const cudaDeviceProp *props,
          cudaStream_t stream);
} // namespace cuda
} // namespace arrayfire
