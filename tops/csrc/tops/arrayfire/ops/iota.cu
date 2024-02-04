/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include "tops/arrayfire/common/array.h"
#include "tops/arrayfire/common/mem.h"
#include <cuda_fp16.h>
// #include <common/half.hpp>
// #include <err_cuda.hpp>
#include "tops/arrayfire/kernel/iota.h"
#include "tops/arrayfire/ops/iota.h"
// #include <iota.hpp>
// #include <iota.hpp>
// #include <kernel/iota.hpp>
// #include <math.hpp>
#include <stdexcept>

// using arrayfire::common::half;

namespace arrayfire {
namespace cuda {
template <typename T>
Array<T> iota(const cudaDeviceProp *props, cudaStream_t stream,
              const arrayfire::dim4 &dims, const arrayfire::dim4 &tile_dims) {
  dim4 outdims = dims * tile_dims;

  Array<T> out = arrayfire::createEmptyArray<T>(outdims);
  arrayfire::cuda::kernel::iota<T>(out.GetKernelParams(), dims, props, stream);

  return out;
}

#define INSTANTIATE(T)                                                         \
  template Array<T> iota<T>(const cudaDeviceProp *props, cudaStream_t stream,  \
                            const arrayfire::dim4 &dims,                       \
                            const arrayfire::dim4 &tile_dims);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint32_t)
// INSTANTIATE(intl)
// INSTANTIATE(uintl)
// INSTANTIATE(uchar)
// INSTANTIATE(short)
// INSTANTIATE(ushort)
// INSTANTIATE(__half)
} // namespace cuda
} // namespace arrayfire
