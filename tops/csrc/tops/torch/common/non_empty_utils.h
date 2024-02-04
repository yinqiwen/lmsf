// #include <ATen/core/TensorBase.h>
#include <algorithm>
#include <vector>
#include "tops/c_api/c_api.h"

namespace at::native {

inline int64_t ensure_nonempty_dim(int64_t dim) { return std::max<int64_t>(dim, 1); }

inline int64_t ensure_nonempty_size(const CTensorView &t, int64_t dim) { return t.ndim == 0 ? 1 : t.shape[dim]; }

inline int64_t ensure_nonempty_stride(const CTensorView &t, int64_t dim) { return t.ndim == 0 ? 1 : t.stride[dim]; }

using IdxVec = std::vector<int64_t>;
inline IdxVec ensure_nonempty_vec(IdxVec vec) {
  if (vec.empty()) {
    vec.push_back(1);
  }
  return vec;
}

}  // namespace at::native
