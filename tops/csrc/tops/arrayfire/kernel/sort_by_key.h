/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

// #include <Param.hpp>
// #include <common/dispatch.hpp>
// #include <debug_cuda.hpp>
// #include <err_cuda.hpp>
// #include <iota.hpp>
// #include <kernel/thrust_sort_by_key.hpp>
// #include <math.hpp>
// #include <memory.hpp>
// #include <af/dim4.hpp>
#include "tops/arrayfire/common/dim4.h"
#include "tops/arrayfire/common/mem.h"
#include "tops/arrayfire/common/param.h"
#include "tops/arrayfire/kernel/thrust_sort_by_key_impl.h"
#include "tops/arrayfire/ops/iota.h"
#include "tops/common/err_cuda.h"

namespace arrayfire {
namespace cuda {
namespace kernel {
// Wrapper functions
template <typename Tk, typename Tv>
void sort0ByKeyIterative(Param<Tk> okey, Param<Tv> oval, bool isAscending) {
  for (int w = 0; w < okey.dims[3]; w++) {
    int okeyW = w * okey.strides[3];
    int ovalW = w * oval.strides[3];
    for (int z = 0; z < okey.dims[2]; z++) {
      int okeyWZ = okeyW + z * okey.strides[2];
      int ovalWZ = ovalW + z * oval.strides[2];
      for (int y = 0; y < okey.dims[1]; y++) {
        int okeyOffset = okeyWZ + y * okey.strides[1];
        int ovalOffset = ovalWZ + y * oval.strides[1];

        thrustSortByKey<Tk, Tv>(okey.ptr + okeyOffset, oval.ptr + ovalOffset,
                                okey.dims[0], isAscending);
      }
    }
  }
  // POST_LAUNCH_CHECK();
}

template <typename Tk, typename Tv>
void sortByKeyBatched(Param<Tk> pKey, Param<Tv> pVal, const int dim,
                      bool isAscending, const cudaDeviceProp *props,
                      cudaStream_t stream) {
  arrayfire::dim4 inDims;
  for (int i = 0; i < 4; i++)
    inDims[i] = pKey.dims[i];

  const dim_t elements = inDims.elements();

  // Sort dimension
  // tileDims * seqDims = inDims
  arrayfire::dim4 tileDims(1);
  arrayfire::dim4 seqDims = inDims;
  tileDims[dim] = inDims[dim];
  seqDims[dim] = 1;

  // Create/call iota
  Array<uint32_t> Seq = iota<uint32_t>(props, stream, seqDims, tileDims);

  Tk *Key = pKey.ptr;

  // Tk* ckey = nullptr;
  // cudaMalloc(&ckey, sizeof(Tk) * elements);
  // auto cKey = memAlloc<Tk>(elements);
  auto ckey = arrayfire::device_alloc<Tk>(elements);
  CUDA_CHECK(cudaMemcpyAsync(ckey.get(), Key, elements * sizeof(Tk),
                             cudaMemcpyDeviceToDevice, stream));

  Tv *Val = pVal.ptr;
  thrustSortByKey(Key, Val, elements, isAscending);
  thrustSortByKey(ckey.get(), Seq.get(), elements, isAscending);

  // auto cSeq = memAlloc<uint>(elements);
  auto cSeq = arrayfire::device_alloc<uint32_t>(elements);
  CUDA_CHECK(cudaMemcpyAsync(cSeq.get(), Seq.get(), elements * sizeof(uint),
                             cudaMemcpyDeviceToDevice, stream));

  // This always needs to be ascending
  thrustSortByKey(Seq.get(), Val, elements, true);
  thrustSortByKey(cSeq.get(), Key, elements, true);

  // No need of doing moddims here because the original Array<T>
  // dimensions have not been changed
  // val.modDims(inDims);
}

template <typename Tk, typename Tv>
void sort0ByKey(Param<Tk> okey, Param<Tv> oval, bool isAscending,
                const cudaDeviceProp *props, cudaStream_t stream) {
  int higherDims = okey.dims[1] * okey.dims[2] * okey.dims[3];

  // Batced sort performs 4x sort by keys But this is only useful
  // before GPU is saturated The GPU is saturated at around 100,000
  // integers Call batched sort only if both conditions are met
  if (higherDims > 4 && okey.dims[0] < 100000)
    sortByKeyBatched<Tk, Tv>(okey, oval, 0, isAscending, props, stream);
  else
    sort0ByKeyIterative<Tk, Tv>(okey, oval, isAscending);
}
} // namespace kernel
} // namespace cuda
} // namespace arrayfire
