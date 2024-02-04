/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// #include "config.hpp"
// #include <Param.hpp>
// #include <backend.hpp>
// #include <common/dispatch.hpp>
// #include <common/kernel_cache.hpp>
// #include <debug_cuda.hpp>
// #include <err_cuda.hpp>
// #include <memory.hpp>
// #include <nvrtc_kernel_headers/scan_dim_cuh.hpp>
#include "tops/arrayfire/common/mem.h"
#include "tops/arrayfire/common/optypes.h"
#include "tops/arrayfire/common/param.h"
#include "tops/arrayfire/kernel/config.h"
#include "tops/arrayfire/kernel/scan_dim.cuh"
#include <stdexcept>

namespace arrayfire {
namespace cuda {
namespace kernel {

template <typename Ti, typename To, af_op_t op>
static void
scan_dim_launcher(Param<To> out, Param<To> tmp, CParam<Ti> in,
                  const uint32_t threads_y, const dim_t blocks_all[4], int dim,
                  bool isFinalPass, bool inclusive_scan,
                  const cudaDeviceProp *props, cudaStream_t stream) {
  // auto scan_dim = common::getKernel(
  //     "arrayfire::cuda::scan_dim", {{scan_dim_cuh_src}},
  //     TemplateArgs(TemplateTypename<Ti>(), TemplateTypename<To>(),
  //                  TemplateArg(op), TemplateArg(dim),
  //                  TemplateArg(isFinalPass), TemplateArg(threads_y),
  //                  TemplateArg(inclusive_scan)),
  //     {{DefineValue(THREADS_X)}});

  dim3 threads(THREADS_X, threads_y);

  dim3 blocks(blocks_all[0] * blocks_all[2], blocks_all[1] * blocks_all[3]);

  const int maxBlocksY = props->maxGridSize[1];
  blocks.z = divup(blocks.y, maxBlocksY);
  blocks.y = divup(blocks.y, blocks.z);

  uint lim = divup(out.dims[dim], (threads_y * blocks_all[dim]));

  // EnqueueArgs qArgs(blocks, threads, getActiveStream());
  // scan_dim(qArgs, out, tmp, in, blocks_all[0], blocks_all[1],
  // blocks_all[dim], lim); POST_LAUNCH_CHECK();
  switch (threads_y) {
  case 8: {
    arrayfire::cuda::scan_dim_kernel<Ti, To, op, 8>
        <<<blocks, threads, 0, stream>>>(out, tmp, in, blocks_all[0],
                                         blocks_all[1], blocks_all[dim], lim,
                                         dim, isFinalPass, inclusive_scan);
    break;
  }
  case 4: {
    arrayfire::cuda::scan_dim_kernel<Ti, To, op, 4>
        <<<blocks, threads, 0, stream>>>(out, tmp, in, blocks_all[0],
                                         blocks_all[1], blocks_all[dim], lim,
                                         dim, isFinalPass, inclusive_scan);
    break;
  }
  case 2: {
    arrayfire::cuda::scan_dim_kernel<Ti, To, op, 2>
        <<<blocks, threads, 0, stream>>>(out, tmp, in, blocks_all[0],
                                         blocks_all[1], blocks_all[dim], lim,
                                         dim, isFinalPass, inclusive_scan);
    break;
  }
  case 1: {
    arrayfire::cuda::scan_dim_kernel<Ti, To, op, 1>
        <<<blocks, threads, 0, stream>>>(out, tmp, in, blocks_all[0],
                                         blocks_all[1], blocks_all[dim], lim,
                                         dim, isFinalPass, inclusive_scan);
    break;
  }
  default: {
    printf("unsupported thread_y:%u for scan_dim_launcher\n", threads_y);
    throw new std::runtime_error("unsupported thread_y for scan_dim_launche");
  }
  }
}

template <typename To, af_op_t op>
static void
bcast_dim_launcher(Param<To> out, CParam<To> tmp, const uint32_t threads_y,
                   const dim_t blocks_all[4], int dim, bool inclusive_scan,
                   const cudaDeviceProp *props, cudaStream_t stream) {
  // auto scan_dim_bcast = common::getKernel("arrayfire::cuda::scan_dim_bcast",
  // {{scan_dim_cuh_src}},
  //     TemplateArgs(TemplateTypename<To>(), TemplateArg(op),
  //     TemplateArg(dim)));

  dim3 threads(THREADS_X, threads_y);

  dim3 blocks(blocks_all[0] * blocks_all[2], blocks_all[1] * blocks_all[3]);

  const int maxBlocksY = props->maxGridSize[1];
  blocks.z = divup(blocks.y, maxBlocksY);
  blocks.y = divup(blocks.y, blocks.z);

  uint32_t lim = divup(out.dims[dim], (threads_y * blocks_all[dim]));

  // EnqueueArgs qArgs(blocks, threads, getActiveStream());
  // scan_dim_bcast(qArgs, out, tmp, blocks_all[0], blocks_all[1],
  // blocks_all[dim], lim, inclusive_scan); POST_LAUNCH_CHECK();
  arrayfire::cuda::scan_dim_bcast_kernel<To, op>
      <<<blocks, threads, 0, stream>>>(out, tmp, blocks_all[0], blocks_all[1],
                                       blocks_all[dim], lim, inclusive_scan,
                                       dim);
}

template <typename Ti, typename To, af_op_t op>
static void scan_dim(Param<To> out, CParam<Ti> in, int dim, bool inclusive_scan,
                     const cudaDeviceProp *props, cudaStream_t stream) {
  uint threads_y =
      std::min(THREADS_Y, arrayfire::cuda::nextpow2(out.dims[dim]));
  uint threads_x = THREADS_X;

  dim_t blocks_all[] = {divup(out.dims[0], threads_x), out.dims[1], out.dims[2],
                        out.dims[3]};

  blocks_all[dim] = divup(out.dims[dim], threads_y * REPEAT);

  if (blocks_all[dim] == 1) {
    scan_dim_launcher<Ti, To, op>(out, out, in, threads_y, blocks_all, dim,
                                  true, inclusive_scan, props, stream);
  } else {
    Param<To> tmp = out;

    tmp.dims[dim] = blocks_all[dim];
    tmp.strides[0] = 1;
    for (int k = 1; k < 4; k++)
      tmp.strides[k] = tmp.strides[k - 1] * tmp.dims[k - 1];

    int tmp_elements = tmp.strides[3] * tmp.dims[3];
    // auto tmp_alloc = memAlloc<To>(tmp_elements);
    auto tmp_alloc = arrayfire::device_alloc<To>(tmp_elements);
    tmp.ptr = tmp_alloc.get();

    scan_dim_launcher<Ti, To, op>(out, tmp, in, threads_y, blocks_all, dim,
                                  false, inclusive_scan, props, stream);

    int bdim = blocks_all[dim];
    blocks_all[dim] = 1;

    // FIXME: Is there an alternative to the if condition ?
    if (op == af_notzero_t) {
      scan_dim_launcher<To, To, af_add_t>(tmp, tmp, tmp, threads_y, blocks_all,
                                          dim, true, true, props, stream);
    } else {
      scan_dim_launcher<To, To, op>(tmp, tmp, tmp, threads_y, blocks_all, dim,
                                    true, true, props, stream);
    }

    blocks_all[dim] = bdim;
    bcast_dim_launcher<To, op>(out, tmp, threads_y, blocks_all, dim,
                               inclusive_scan, props, stream);
  }
}

} // namespace kernel
} // namespace cuda
} // namespace arrayfire
