/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
// #include <ThrustArrayFirePolicy.hpp>
// #include <ThrustAllocator.cuh>
#include <thrust/memory.h>
#include <thrust/system/cuda/detail/par.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/version.h>

namespace arrayfire
{
namespace cuda
{
// template <typename T>
// using ThrustVector = thrust::device_vector<T, ThrustAllocator<T>>;

struct ThrustArrayFirePolicy : thrust::cuda::execution_policy<ThrustArrayFirePolicy>
{
};
} // namespace cuda
} // namespace arrayfire

#define THRUST_SELECT(fn, ...) fn(arrayfire::cuda::ThrustArrayFirePolicy(), __VA_ARGS__)
#define THRUST_SELECT_OUT(res, fn, ...) res = fn(arrayfire::cuda::ThrustArrayFirePolicy(), __VA_ARGS__)
