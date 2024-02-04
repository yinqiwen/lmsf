/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// #include <Array.hpp>
// #include <copy.hpp>
// #include <err_cuda.hpp>
// #include <kernel/sort_by_key.hpp>
// #include <math.hpp>
// #include <range.hpp>
// #include <reorder.hpp>
// #include <sort_index.hpp>
#include "tops/arrayfire/common/dim4.h"
#include "tops/arrayfire/kernel/sort_by_key.h"
#include "tops/arrayfire/ops/range.h"
#include "tops/arrayfire/ops/reorder.h"
#include <cuda_fp16.h>
#include <stdexcept>

namespace arrayfire {
namespace cuda {
template <typename T>
void sort_index(Array<T> &okey, Array<uint32_t> &oval, const Array<T> &in,
                const uint32_t dim, bool isAscending,
                const cudaDeviceProp *props, cudaStream_t stream) {
  if (okey.is_empty()) {
    okey = arrayfire::copyArray<T>(in, stream);
  }
  arrayfire::cuda::range<uint32_t>(props, stream, in.dims(), oval, dim);

  // printf("indices prepare:%s\n", oval.to_string().c_str());

  switch (dim) {
  case 0:
    kernel::sort0ByKey<T, uint32_t>(okey.GetKernelParams(),
                                    oval.GetKernelParams(), isAscending, props,
                                    stream);
    break;
  case 1:
  case 2:
  case 3:
    kernel::sortByKeyBatched<T, uint32_t>(okey.GetKernelParams(),
                                          oval.GetKernelParams(), dim,
                                          isAscending, props, stream);
    break;
  default: {
    throw new std::runtime_error("not supported dims for sror_index");
  }
  }

  if (dim != 0) {
    arrayfire::dim4 preorderDims = okey.dims();
    arrayfire::dim4 reorderDims(0, 1, 2, 3);
    reorderDims[dim] = 0;
    preorderDims[0] = okey.dims()[dim];
    for (int i = 1; i <= (int)dim; i++) {
      reorderDims[i - 1] = i;
      preorderDims[i] = okey.dims()[i - 1];
    }

    okey.setDataDims(preorderDims);
    oval.setDataDims(preorderDims);

    // okey = reorder<T>(okey, reorderDims);
    // oval = reorder<uint>(oval, reorderDims);

    okey = arrayfire::cuda::reorder<T>(okey, reorderDims, props, stream);
    oval = arrayfire::cuda::reorder<uint32_t>(oval, reorderDims, props, stream);
  }
}

#define INSTANTIATE(T)                                                         \
  template void sort_index<T>(Array<T> & val, Array<uint32_t> & idx,           \
                              const Array<T> &in, const uint dim,              \
                              bool isAscending, const cudaDeviceProp *props,   \
                              cudaStream_t stream);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint32_t)
// INSTANTIATE(char)
// INSTANTIATE(uchar)
// INSTANTIATE(short)
// INSTANTIATE(ushort)
// INSTANTIATE(intl)
// INSTANTIATE(uintl)
INSTANTIATE(__half)

} // namespace cuda
} // namespace arrayfire
