/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include "tops/arrayfire/common/mem.h"
#include "tops/arrayfire/kernel/scan_dim.h"
#include "tops/arrayfire/kernel/scan_first.h"
#include "tops/arrayfire/ops/scan.h"
#include <cuda_fp16.h>

#include <stdexcept>

namespace arrayfire {
namespace cuda {
template <af_op_t op, typename Ti, typename To>
void scan(Array<To> &out, const Array<Ti> &in, const int dim,
          bool inclusive_scan, const cudaDeviceProp *props,
          cudaStream_t stream) {
  if (out.is_empty()) {
    out = arrayfire::createEmptyArray<To>(in.dims());
  }
  // Array<To> out = createEmptyArray<To>(in.dims());

  if (dim == 0) {
    kernel::scan_first<Ti, To, op>(out.GetKernelParams(), in.GetKernelParams(),
                                   inclusive_scan, props, stream);
  } else {
    kernel::scan_dim<Ti, To, op>(out.GetKernelParams(), in.GetKernelParams(),
                                 dim, inclusive_scan, props, stream);
  }
}

#define INSTANTIATE_SCAN(ROp, Ti, To)                                          \
  template void scan<ROp, Ti, To>(                                             \
      Array<To> & out, const Array<Ti> &in, const int dim,                     \
      bool inclusive_scan, const cudaDeviceProp *props, cudaStream_t stream);

#define INSTANTIATE_SCAN_ALL(ROp)                                              \
  INSTANTIATE_SCAN(ROp, float, float)                                          \
  INSTANTIATE_SCAN(ROp, double, double)                                        \
  INSTANTIATE_SCAN(ROp, int, int)                                              \
  INSTANTIATE_SCAN(ROp, uint32_t, uint32_t)                                    \
  INSTANTIATE_SCAN(ROp, int64_t, int64_t)                                      \
  INSTANTIATE_SCAN(ROp, __half, __half)

INSTANTIATE_SCAN(af_notzero_t, char, uint32_t)
INSTANTIATE_SCAN_ALL(af_add_t)
INSTANTIATE_SCAN_ALL(af_mul_t)
INSTANTIATE_SCAN_ALL(af_min_t)
INSTANTIATE_SCAN_ALL(af_max_t)
} // namespace cuda
} // namespace arrayfire
