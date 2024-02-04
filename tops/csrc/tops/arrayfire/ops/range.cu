/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// #include <range.hpp>

// #include <Array.hpp>
// #include <err_cuda.hpp>
// #include <kernel/range.hpp>
// #include <math.hpp>
#include "tops/arrayfire/common/mem.h"
#include "tops/arrayfire/kernel/range.h"
#include <stdexcept>

// using arrayfire::common::half;

namespace arrayfire {
namespace cuda {
template <typename T>
void range(const cudaDeviceProp *props, cudaStream_t stream,
           const arrayfire::dim4 &dim, Array<T> &out, const int seq_dim) {
  // Set dimension along which the sequence should be
  // Other dimensions are simply tiled
  int _seq_dim = seq_dim;
  if (seq_dim < 0) {
    _seq_dim = 0; // column wise sequence
  }
  // printf("####_seq_dim:%d\n",_seq_dim);
  if (_seq_dim < 0 || _seq_dim > 3) {
    throw new std::runtime_error("Invalid rep selection");
    // AF_ERROR("Invalid rep selection", AF_ERR_ARG);
  }
  if (out.is_empty()) {
    out = arrayfire::createEmptyArray<T>(dim);
  }
  // Array<T> out = arrayfire::createEmptyArray<T>(dim);
  // Array<T> out = createEmptyArray<T>(dim);
  arrayfire::cuda::kernel::range<T>(out.GetKernelParams(), _seq_dim, props,
                                    stream);

  // return out;
}

#define INSTANTIATE(T)                                                         \
  template void range<T>(const cudaDeviceProp *props, cudaStream_t stream,     \
                         const arrayfire::dim4 &dim, Array<T> &out,            \
                         const int seq_dim);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint32_t)
// INSTANTIATE(intl)
// INSTANTIATE(uintl)
// INSTANTIATE(uchar)
// INSTANTIATE(short)
// INSTANTIATE(ushort)
// INSTANTIATE(half)
} // namespace cuda
} // namespace arrayfire
