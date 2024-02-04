#pragma once

#include "tops/arrayfire/common/types.h"
#include <algorithm>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <limits>
#include <math_constants.h>

namespace arrayfire {
namespace cuda {
#ifdef AF_WITH_FAST_MATH
constexpr bool fast_math = true;
#else
constexpr bool fast_math = false;
#endif

template <typename T> static inline __DH__ T abs(T val) { return ::abs(val); }

static inline __DH__ int abs(int val) { return (val > 0 ? val : -val); }

static inline __DH__ char abs(char val) { return (val > 0 ? val : -val); }

static inline __DH__ float abs(float val) { return fabsf(val); }

static inline __DH__ double abs(double val) { return fabs(val); }

static inline __DH__ float abs(cfloat cval) { return cuCabsf(cval); }

static inline __DH__ double abs(cdouble cval) { return cuCabs(cval); }

static inline __DH__ size_t min(size_t lhs, size_t rhs) {
  return lhs < rhs ? lhs : rhs;
}

static inline __DH__ size_t max(size_t lhs, size_t rhs) {
  return lhs > rhs ? lhs : rhs;
}

static inline __device__ __half abs(__half val) {
  return __short_as_half(__half_as_short(val) & 0x7FFF);
}

template <typename T> inline __DH__ T min(T lhs, T rhs) {
  return ::min(lhs, rhs);
}

template <typename T> inline __DH__ T max(T lhs, T rhs) {
  return ::max(lhs, rhs);
}

template <> inline __DH__ __half min<__half>(__half lhs, __half rhs) {
#if __CUDA_ARCH__ >= 530
  return __hlt(lhs, rhs) ? lhs : rhs;
#else
  return __half2float(lhs) < __half2float(rhs) ? lhs : rhs;
#endif
}

template <> inline __DH__ __half max<__half>(__half lhs, __half rhs) {
#if __CUDA_ARCH__ >= 530
  return __hgt(lhs, rhs) ? lhs : rhs;
#else
  return __half2float(lhs) > __half2float(rhs) ? lhs : rhs;
#endif
}

template <> __DH__ inline cfloat max<cfloat>(cfloat lhs, cfloat rhs) {
  return abs(lhs) > abs(rhs) ? lhs : rhs;
}

template <> __DH__ inline cdouble max<cdouble>(cdouble lhs, cdouble rhs) {
  return abs(lhs) > abs(rhs) ? lhs : rhs;
}

template <> __DH__ inline cfloat min<cfloat>(cfloat lhs, cfloat rhs) {
  return abs(lhs) < abs(rhs) ? lhs : rhs;
}

template <> __DH__ inline cdouble min<cdouble>(cdouble lhs, cdouble rhs) {
  return abs(lhs) < abs(rhs) ? lhs : rhs;
}

template <typename T> __DH__ static T scalar(double val) { return (T)(val); }

template <> __DH__ inline cfloat scalar<cfloat>(double val) {
  cfloat cval = {(float)val, 0};
  return cval;
}

template <> __DH__ inline cdouble scalar<cdouble>(double val) {
  cdouble cval = {val, 0};
  return cval;
}

template <typename To, typename Ti> __DH__ static To scalar(Ti real, Ti imag) {
  To cval = {real, imag};
  return cval;
}

template <typename T> inline __device__ T maxval() {
  return 1u << (8 * sizeof(T) - 1);
}

template <typename T> inline __device__ T minval() { return scalar<T>(0); }

template <> inline __device__ int maxval<int>() { return 0x7fffffff; }

template <> inline __device__ int minval<int>() { return 0x80000000; }

template <> inline __device__ int64_t maxval<int64_t>() {
  return 0x7fffffffffffffff;
}

template <> inline __device__ int64_t minval<int64_t>() {
  return 0x8000000000000000;
}

template <> inline __device__ uint64_t maxval<uint64_t>() {
  return 1ULL << (8 * sizeof(uint64_t) - 1);
}

template <> inline __device__ char maxval<char>() { return 0x7f; }

template <> inline __device__ char minval<char>() { return 0x80; }

template <> inline __device__ float maxval<float>() { return CUDART_INF_F; }

template <> inline __device__ float minval<float>() { return -CUDART_INF_F; }

template <> inline __device__ double maxval<double>() { return CUDART_INF; }

template <> inline __device__ double minval<double>() { return -CUDART_INF; }

template <> inline __device__ short maxval<short>() { return 0x7fff; }

template <> inline __device__ short minval<short>() { return 0x8000; }

template <> inline __device__ ushort maxval<ushort>() {
  return ((ushort)1) << (8 * sizeof(ushort) - 1);
}

// template <>
// inline __device__ common::half maxval<common::half>()
// {
//     return common::half(65537.f);
// }

// template <>
// inline __device__ common::half minval<common::half>()
// {
//     return common::half(-65537.f);
// }

template <> inline __device__ __half maxval<__half>() {
  return __float2half(CUDART_INF);
}

template <> inline __device__ __half minval<__half>() {
  return __float2half(-CUDART_INF);
}

#define upcast cuComplexFloatToDouble
#define downcast cuComplexDoubleToFloat

#ifdef __GNUC__
// This suprresses unused function warnings in gcc
// FIXME: Check if the warnings exist in other compilers
#define __SDH__ static __DH__ __attribute__((unused))
#else
#define __SDH__ static __DH__
#endif
__SDH__ float real(cfloat c) { return cuCrealf(c); }

__SDH__ double real(cdouble c) { return cuCreal(c); }

__SDH__ float imag(cfloat c) { return cuCimagf(c); }

__SDH__ double imag(cdouble c) { return cuCimag(c); }

template <typename T> static inline __DH__ auto is_nan(const T &val) -> bool {
  return false;
}

template <> inline __DH__ auto is_nan<float>(const float &val) -> bool {
  return ::isnan(val);
}

template <> inline __DH__ auto is_nan<double>(const double &val) -> bool {
  return ::isnan(val);
}

#ifdef __CUDA_ARCH__
template <> inline __device__ auto is_nan<__half>(const __half &val) -> bool {
#if __CUDA_ARCH__ >= 530
  return __hisnan(val);
#else
  return ::isnan(__half2float(val));
#endif
}
#endif

template <> inline auto is_nan<cfloat>(const cfloat &in) -> bool {
  return ::isnan(real(in)) || ::isnan(imag(in));
}

template <> inline auto is_nan<cdouble>(const cdouble &in) -> bool {
  return ::isnan(real(in)) || ::isnan(imag(in));
}

template <typename T> T __SDH__ conj(T x) { return x; }

__SDH__ cfloat conj(cfloat c) { return cuConjf(c); }

__SDH__ cdouble conj(cdouble c) { return cuConj(c); }

__SDH__ cfloat make_cfloat(bool x) {
  return make_cuComplex(static_cast<float>(x), 0);
}

__SDH__ cfloat make_cfloat(int x) {
  return make_cuComplex(static_cast<float>(x), 0);
}

__SDH__ cfloat make_cfloat(unsigned x) {
  return make_cuComplex(static_cast<float>(x), 0);
}

__SDH__ cfloat make_cfloat(short x) {
  return make_cuComplex(static_cast<float>(x), 0);
}

__SDH__ cfloat make_cfloat(ushort x) {
  return make_cuComplex(static_cast<float>(x), 0);
}

__SDH__ cfloat make_cfloat(float x) {
  return make_cuComplex(static_cast<float>(x), 0);
}

__SDH__ cfloat make_cfloat(double x) {
  return make_cuComplex(static_cast<float>(x), 0);
}

__SDH__ cfloat make_cfloat(cfloat x) { return x; }

__SDH__ cfloat make_cfloat(cdouble c) { return make_cuComplex(c.x, c.y); }

__SDH__ cdouble make_cdouble(bool x) {
  return make_cuDoubleComplex(static_cast<double>(x), 0);
}

__SDH__ cdouble make_cdouble(int x) {
  return make_cuDoubleComplex(static_cast<double>(x), 0);
}

__SDH__ cdouble make_cdouble(unsigned x) {
  return make_cuDoubleComplex(static_cast<double>(x), 0);
}

__SDH__ cdouble make_cdouble(short x) {
  return make_cuDoubleComplex(static_cast<double>(x), 0);
}

__SDH__ cdouble make_cdouble(ushort x) {
  return make_cuDoubleComplex(static_cast<double>(x), 0);
}

__SDH__ cdouble make_cdouble(float x) {
  return make_cuDoubleComplex(static_cast<double>(x), 0);
}

__SDH__ cdouble make_cdouble(double x) {
  return make_cuDoubleComplex(static_cast<double>(x), 0);
}

__SDH__ cdouble make_cdouble(cdouble x) { return x; }

__SDH__ cdouble make_cdouble(cfloat c) {
  return make_cuDoubleComplex(static_cast<double>(c.x), c.y);
}

__SDH__ cfloat make_cfloat(float x, float y) { return make_cuComplex(x, y); }

__SDH__ cdouble make_cdouble(double x, double y) {
  return make_cuDoubleComplex(x, y);
}

inline unsigned nextpow2(unsigned x) {
  x = x - 1U;
  x = x | (x >> 1U);
  x = x | (x >> 2U);
  x = x | (x >> 4U);
  x = x | (x >> 8U);
  x = x | (x >> 16U);
  return x + 1U;
}

} // namespace cuda
} // namespace arrayfire