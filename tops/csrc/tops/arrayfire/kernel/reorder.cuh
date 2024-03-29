/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

// #include <Param.hpp>
#include "tops/arrayfire/common/param.h"

namespace arrayfire {
namespace cuda {

template <typename T>
__global__ void reorder_kernel(Param<T> out, CParam<T> in, const int d0,
                               const int d1, const int d2, const int d3,
                               const int blocksPerMatX,
                               const int blocksPerMatY) {
  const int oz = blockIdx.x / blocksPerMatX;
  const int ow = (blockIdx.y + blockIdx.z * gridDim.y) / blocksPerMatY;

  const int blockIdx_x = blockIdx.x - oz * blocksPerMatX;
  const int blockIdx_y =
      (blockIdx.y + blockIdx.z * gridDim.y) - ow * blocksPerMatY;

  const int xx = threadIdx.x + blockIdx_x * blockDim.x;
  const int yy = threadIdx.y + blockIdx_y * blockDim.y;

  if (xx >= out.dims[0] || yy >= out.dims[1] || oz >= out.dims[2] ||
      ow >= out.dims[3])
    return;

  const int incy = blocksPerMatY * blockDim.y;
  const int incx = blocksPerMatX * blockDim.x;

  const int rdims[] = {d0, d1, d2, d3};
  const int o_off = ow * out.strides[3] + oz * out.strides[2];
  int ids[4] = {0};
  ids[rdims[3]] = ow;
  ids[rdims[2]] = oz;

  for (int oy = yy; oy < out.dims[1]; oy += incy) {
    ids[rdims[1]] = oy;
    for (int ox = xx; ox < out.dims[0]; ox += incx) {
      ids[rdims[0]] = ox;

      const int oIdx = o_off + oy * out.strides[1] + ox;

      const int iIdx = ids[3] * in.strides[3] + ids[2] * in.strides[2] +
                       ids[1] * in.strides[1] + ids[0];

      out.ptr[oIdx] = in.ptr[iIdx];
    }
  }
}

} // namespace cuda
} // namespace arrayfire
