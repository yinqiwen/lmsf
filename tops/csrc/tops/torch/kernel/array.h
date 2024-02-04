#pragma once

// A fixed-size array type usable from both host and
// device code.

// #include <c10/macros/Macros.h>
// #include <c10/util/irange.h>

namespace at {
namespace detail {

template <typename T, int size_>
struct Array {
  T data[size_];

  __host__ __device__ T operator[](int i) const { return data[i]; }
  __host__ __device__ T& operator[](int i) { return data[i]; }
#if defined(USE_ROCM)
  C10_HOST_DEVICE Array() = default;
  C10_HOST_DEVICE Array(const Array&) = default;
  C10_HOST_DEVICE Array& operator=(const Array&) = default;
#else
  Array() = default;
  Array(const Array&) = default;
  Array& operator=(const Array&) = default;
#endif
  static constexpr int size() { return size_; }
  // Fill the array with x.
  __host__ __device__ Array(T x) {
    for (int i = 0; i < size_; i++) {
      data[i] = x;
    }
  }
};

}  // namespace detail
}  // namespace at
