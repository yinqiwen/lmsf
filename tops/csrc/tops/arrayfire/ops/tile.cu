/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// #include <tile.hpp>

// #include <Array.hpp>
// #include <common/half.hpp>
// #include <err_cuda.hpp>
// #include <kernel/tile.hpp>
#include <cuda_fp16.h>
#include "tops/arrayfire/common/mem.h"
#include "tops/arrayfire/kernel/tile.h"
#include "tops/arrayfire/ops/tile.h"

#include <stdexcept>

// using arrayfire::common::half;

namespace arrayfire {
namespace cuda {
template <typename T>
void tile(const Array<T> &in, const arrayfire::dim4 &tileDims, const cudaDeviceProp *props, cudaStream_t stream,
          Array<T> &out) {
  const arrayfire::dim4 &iDims = in.dims();
  arrayfire::dim4 oDims = iDims;
  oDims *= tileDims;
  if (out.is_empty()) {
    out = arrayfire::createEmptyArray<T>(oDims);
  }
  arrayfire::cuda::kernel::tile(out.GetKernelParams(), in.GetKernelParams(), props, stream);
  //   if (iDims.elements() == 0 || oDims.elements() == 0) {
  //     // AF_ERROR("Elements are 0", AF_ERR_SIZE);
  //   }

  //   Array<T> out = createEmptyArray<T>(oDims);

  //   kernel::tile<T>(out, in);

  //   return out;
}

#define INSTANTIATE(T)                                                                                    \
  template void tile<T>(const Array<T> &in, const arrayfire::dim4 &tileDims, const cudaDeviceProp *props, \
                        cudaStream_t stream, Array<T> &out);

INSTANTIATE(float)
INSTANTIATE(double)
// INSTANTIATE(cfloat)
// INSTANTIATE(cdouble)
// INSTANTIATE(int)
// INSTANTIATE(uint)
// INSTANTIATE(intl)
// INSTANTIATE(uintl)
// INSTANTIATE(uchar)
// INSTANTIATE(char)
// INSTANTIATE(short)
// INSTANTIATE(ushort)
INSTANTIATE(half)

}  // namespace cuda
}  // namespace arrayfire
