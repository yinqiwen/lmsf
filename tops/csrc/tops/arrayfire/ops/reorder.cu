/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// #include <reorder.hpp>

// #include <Array.hpp>
// #include <common/half.hpp>
// #include <err_cuda.hpp>
// #include <kernel/reorder.hpp>
#include "tops/arrayfire/common/mem.h"
#include "tops/arrayfire/kernel/reorder.h"
#include "tops/arrayfire/ops/reorder.h"
#include <cuda_fp16.h>

#include <stdexcept>

// using arrayfire::common::half;

namespace arrayfire {
namespace cuda {

template <typename T>
Array<T> reorder(const Array<T> &in, const arrayfire::dim4 &rdims,
                 const cudaDeviceProp *props, cudaStream_t stream) {
  const arrayfire::dim4 &iDims = in.dims();
  arrayfire::dim4 oDims(0);
  for (int i = 0; i < 4; i++) {
    oDims[i] = iDims[rdims[i]];
  }

  Array<T> out = arrayfire::createEmptyArray<T>(oDims);

  arrayfire::cuda::kernel::reorder<T>(
      out.GetKernelParams(), in.GetKernelParams(), rdims.get(), props, stream);

  return out;
}

#define INSTANTIATE(T)                                                         \
  template Array<T> reorder<T>(                                                \
      const Array<T> &in, const arrayfire::dim4 &rdims,                        \
      const cudaDeviceProp *props, cudaStream_t stream);

INSTANTIATE(float)
INSTANTIATE(double)
// INSTANTIATE(cfloat)
// INSTANTIATE(cdouble)
INSTANTIATE(int)
INSTANTIATE(uint32_t)
// INSTANTIATE(uchar)
// INSTANTIATE(char)
// INSTANTIATE(intl)
// INSTANTIATE(uintl)
// INSTANTIATE(short)
// INSTANTIATE(ushort)
INSTANTIATE(__half)

} // namespace cuda
} // namespace arrayfire
