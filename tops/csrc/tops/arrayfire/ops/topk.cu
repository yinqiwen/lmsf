/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// #include <Array.hpp>
// #include <af/dim4.hpp>
// #include <common/half.hpp>
// #include <kernel/topk.hpp>
// #include <topk.hpp>

#include "tops/arrayfire/common/mem.h"
#include "tops/arrayfire/kernel/topk.cuh"
#include "tops/arrayfire/ops/topk.h"
#include <cuda_fp16.h>

// using arrayfire::common::half;

namespace arrayfire {
namespace cuda {
template <typename T>
void topk(Array<T> &ovals, Array<uint32_t> &oidxs, const Array<T> &ivals,
          const int k, const int dim, const arrayfire::topkFunction order,
          cudaStream_t stream) {
  dim4 outDims = ivals.dims();
  outDims[dim] = k;

  if (ovals.is_empty()) {
    ovals = arrayfire::createEmptyArray<T>(outDims);
  }
  if (oidxs.is_empty()) {
    oidxs = arrayfire::createEmptyArray<uint32_t>(outDims);
  }

  kernel::topk<T>(ovals.GetKernelParams(), oidxs.GetKernelParams(),
                  ivals.GetKernelParams(), k, dim, order, stream);
}

#define INSTANTIATE(T)                                                         \
  template void topk<T>(Array<T> &, Array<uint32_t> &, const Array<T> &,       \
                        const int, const int, const arrayfire::topkFunction,   \
                        cudaStream_t stream);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
// INSTANTIATE(uint)
// INSTANTIATE(long long)
// INSTANTIATE(unsigned long long)
INSTANTIATE(__half)
} // namespace cuda
} // namespace arrayfire
