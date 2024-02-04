/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// #include <Param.hpp>
// #include <backend.hpp>
// #include <common/dispatch.hpp>
// #include <common/kernel_cache.hpp>
// #include <debug_cuda.hpp>
// #include <err_cuda.hpp>
// #include <memory.hpp>
// #include <nvrtc_kernel_headers/scan_first_cuh.hpp>
// #include "config.hpp"

#include "tops/arrayfire/common/mem.h"
#include "tops/arrayfire/common/optypes.h"
#include "tops/arrayfire/common/param.h"
#include "tops/arrayfire/kernel/config.h"
#include "tops/arrayfire/kernel/scan_first.cuh"

namespace arrayfire {
namespace cuda {
namespace kernel {

template <typename Ti, typename To, af_op_t op>
static void scan_first_launcher(
    Param<To> out, Param<To> tmp, CParam<Ti> in, const uint32_t blocks_x,
    const uint32_t blocks_y, const uint32_t threads_x, bool isFinalPass,
    bool inclusive_scan, const cudaDeviceProp *props, cudaStream_t stream) {
  // auto scan_first = common::getKernel("arrayfire::cuda::scan_first",
  // {{scan_first_cuh_src}},
  //     TemplateArgs(TemplateTypename<Ti>(), TemplateTypename<To>(),
  //     TemplateArg(op), TemplateArg(isFinalPass),
  //         TemplateArg(threads_x), TemplateArg(inclusive_scan)),
  //     {{DefineValue(THREADS_PER_BLOCK)}});

  dim3 threads(threads_x, THREADS_PER_BLOCK / threads_x);
  dim3 blocks(blocks_x * out.dims[2], blocks_y * out.dims[3]);

  const int maxBlocksY = props->maxGridSize[1];
  blocks.z = divup(blocks.y, maxBlocksY);
  blocks.y = divup(blocks.y, blocks.z);

  uint lim = divup(out.dims[0], (threads_x * blocks_x));

  // EnqueueArgs qArgs(blocks, threads, stream);
  // scan_first(qArgs, out, tmp, in, blocks_x, blocks_y, lim);
  // POST_LAUNCH_CHECK();

  switch (threads_x) {
  case 256: {
    arrayfire::cuda::scan_first_kernel<Ti, To, op, 256>
        <<<blocks, threads, 0, stream>>>(out, tmp, in, blocks_x, blocks_y, lim,
                                         isFinalPass, inclusive_scan);
    break;
  }
  case 128: {
    arrayfire::cuda::scan_first_kernel<Ti, To, op, 128>
        <<<blocks, threads, 0, stream>>>(out, tmp, in, blocks_x, blocks_y, lim,
                                         isFinalPass, inclusive_scan);
    break;
  }
  case 64: {
    arrayfire::cuda::scan_first_kernel<Ti, To, op, 64>
        <<<blocks, threads, 0, stream>>>(out, tmp, in, blocks_x, blocks_y, lim,
                                         isFinalPass, inclusive_scan);
    break;
  }
  case 32: {
    arrayfire::cuda::scan_first_kernel<Ti, To, op, 32>
        <<<blocks, threads, 0, stream>>>(out, tmp, in, blocks_x, blocks_y, lim,
                                         isFinalPass, inclusive_scan);
    break;
  }
  default: {
    printf("unsupported threads_x:%u for scan_dim_launcher\n", threads_x);
    throw new std::runtime_error(
        "unsupported thread_x for scan_first_launcher");
  }
  }
}

template <typename To, af_op_t op>
static void
bcast_first_launcher(Param<To> out, CParam<To> tmp, const uint32_t blocks_x,
                     const uint32_t blocks_y, const uint32_t threads_x,
                     bool inclusive_scan, const cudaDeviceProp *props,
                     cudaStream_t stream) {
  // auto scan_first_bcast =
  // common::getKernel("arrayfire::cuda::scan_first_bcast",
  // {{scan_first_cuh_src}},
  //     TemplateArgs(TemplateTypename<To>(), TemplateArg(op)));

  dim3 threads(threads_x, THREADS_PER_BLOCK / threads_x);
  dim3 blocks(blocks_x * out.dims[2], blocks_y * out.dims[3]);

  const int maxBlocksY = props->maxGridSize[1];
  blocks.z = divup(blocks.y, maxBlocksY);
  blocks.y = divup(blocks.y, blocks.z);

  uint lim = divup(out.dims[0], (threads_x * blocks_x));

  // EnqueueArgs qArgs(blocks, threads, stream);
  // scan_first_bcast(qArgs, out, tmp, blocks_x, blocks_y, lim, inclusive_scan);
  // POST_LAUNCH_CHECK();
  arrayfire::cuda::scan_first_bcast_kernel<To, op>
      <<<blocks, threads, 0, stream>>>(out, tmp, blocks_x, blocks_y, lim,
                                       inclusive_scan);
}

template <typename Ti, typename To, af_op_t op>
static void scan_first(Param<To> out, CParam<Ti> in, bool inclusive_scan,
                       const cudaDeviceProp *props, cudaStream_t stream) {
  uint32_t threads_x =
      arrayfire::cuda::nextpow2(std::max(32u, (uint)out.dims[0]));
  threads_x = std::min(threads_x, THREADS_PER_BLOCK);
  uint32_t threads_y = THREADS_PER_BLOCK / threads_x;

  uint32_t blocks_x = divup(out.dims[0], threads_x * REPEAT);
  uint32_t blocks_y = divup(out.dims[1], threads_y);

  if (blocks_x == 1) {
    scan_first_launcher<Ti, To, op>(out, out, in, blocks_x, blocks_y, threads_x,
                                    true, inclusive_scan, props, stream);
  } else {
    Param<To> tmp = out;

    tmp.dims[0] = blocks_x;
    tmp.strides[0] = 1;
    for (int k = 1; k < 4; k++)
      tmp.strides[k] = tmp.strides[k - 1] * tmp.dims[k - 1];

    int tmp_elements = tmp.strides[3] * tmp.dims[3];
    // auto tmp_alloc = memAlloc<To>(tmp_elements);
    auto tmp_alloc = arrayfire::device_alloc<To>(tmp_elements);
    tmp.ptr = tmp_alloc.get();

    scan_first_launcher<Ti, To, op>(out, tmp, in, blocks_x, blocks_y, threads_x,
                                    false, inclusive_scan, props, stream);

    // FIXME: Is there an alternative to the if condition ?
    if (op == af_notzero_t) {
      scan_first_launcher<To, To, af_add_t>(
          tmp, tmp, tmp, 1, blocks_y, threads_x, true, true, props, stream);
    } else {
      scan_first_launcher<To, To, op>(tmp, tmp, tmp, 1, blocks_y, threads_x,
                                      true, true, props, stream);
    }

    bcast_first_launcher<To, op>(out, tmp, blocks_x, blocks_y, threads_x,
                                 inclusive_scan, props, stream);
  }
}

} // namespace kernel
} // namespace cuda
} // namespace arrayfire
