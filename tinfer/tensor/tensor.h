/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_fp16.h>
#include <dirent.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_join.h"

#include "tinfer/log/log.h"
#include "tinfer/utils/cleanup.h"
#include "tinfer/utils/cuda_allocator.h"
#include "tinfer/utils/cuda_utils.h"

namespace tinfer {

enum DataType {
  TYPE_INVALID,
  TYPE_BOOL,
  TYPE_UINT8,
  TYPE_UINT16,
  TYPE_UINT32,
  TYPE_UINT64,
  TYPE_INT8,
  TYPE_INT16,
  TYPE_INT32,
  TYPE_INT64,
  TYPE_FP16,
  TYPE_FP32,
  TYPE_FP64,
  TYPE_BYTES,
  TYPE_BF16,
  TYPE_FP8_E4M3,
  TYPE_STR,
  TYPE_VOID,
};

template <typename T>
DataType get_tensor_type() {
  if (std::is_same<T, float>::value || std::is_same<T, const float>::value) {
    return TYPE_FP32;
  } else if (std::is_same<T, half>::value || std::is_same<T, const half>::value) {
    return TYPE_FP16;
  }
#ifdef ENABLE_BF16
  else if (std::is_same<T, __nv_bfloat16>::value || std::is_same<T, const __nv_bfloat16>::value) {
    return TYPE_BF16;
  }
#endif
#ifdef ENABLE_FP8
  else if (std::is_same<T, __nv_fp8_e4m3>::value || std::is_same<T, const __nv_fp8_e4m3>::value) {
    return TYPE_FP8_E4M3;
  }
#endif
  else if (std::is_same<T, int>::value || std::is_same<T, const int>::value) {
    return TYPE_INT32;
  } else if (std::is_same<T, int8_t>::value || std::is_same<T, const int8_t>::value) {
    return TYPE_INT8;
  } else if (std::is_same<T, uint>::value || std::is_same<T, const uint>::value ||
             std::is_same<T, const uint32_t>::value) {
    return TYPE_UINT32;
  } else if (std::is_same<T, unsigned long long int>::value || std::is_same<T, const unsigned long long int>::value) {
    return TYPE_UINT64;
  } else if (std::is_same<T, bool>::value || std::is_same<T, const bool>::value) {
    return TYPE_BOOL;
  } else if (std::is_same<T, char>::value || std::is_same<T, const char>::value) {
    return TYPE_BYTES;
  } else {
    return TYPE_INVALID;
  }
}

enum MemoryType { MEMORY_CPU, MEMORY_CPU_PINNED, MEMORY_GPU };

struct Tensor {
  const MemoryType where;
  const DataType type;
  const std::vector<size_t> shape;
  const void *data;  // TODO(bhseuh) modify from const void* to void* const
  const std::vector<size_t> offsets = std::vector<size_t>{};

  CleanupPtr cleanup_;

  Tensor();
  Tensor(const MemoryType _where, const DataType _type, const std::vector<size_t> _shape, const void *_data);
  Tensor(const MemoryType _where, const DataType _type, const std::vector<size_t> _shape, const void *_data,
         const std::vector<size_t> _offset);

  size_t Size() const;
  size_t SizeBytes() const;

  inline bool Empty() const { return data == nullptr; }

  std::string WhereToString() const;
  std::string ToString() const;
  std::string GetNumpyTypeDesc(DataType type) const;

  void SaveNpy(const std::string &filename) const;
  static Tensor LoadNpy(const std::string &npy_file, const MemoryType where);

  static DataType TypeFromNumpyDesc(std::string type);
  static size_t GetTypeSize(DataType type);

  template <typename T>
  inline T GetVal(size_t index) const {
    CHECK(where == MEMORY_CPU);
    CHECK(data != nullptr);
    CHECK_WITH_INFO(index < Size(), "index is larger than buffer size");

    if (get_tensor_type<T>() != type) {
      TINFER_DEBUG("getVal with type {}, but data type is: {}", GetNumpyTypeDesc(get_tensor_type<T>()),
                   GetNumpyTypeDesc(type));
    }
    return ((T *)data)[index];
  }

  template <typename T>
  inline T GetVal() const {
    // TINFER_DEBUG("{} start", __PRETTY_FUNCTION__);
    if (get_tensor_type<T>() != type) {
      TINFER_DEBUG("getVal with type {}, but data type is: {}", GetNumpyTypeDesc(get_tensor_type<T>()),
                   GetNumpyTypeDesc(type));
    }
    return GetVal<T>(0);
  }

  template <typename T>
  inline T *GetPtr() const {
    if (nullptr == data) {
      return nullptr;
    }
    // FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    if (get_tensor_type<T>() != type) {
      TINFER_DEBUG("getPtr with type {}, but data type is: {}", GetNumpyTypeDesc(get_tensor_type<T>()),
                   GetNumpyTypeDesc(type));
    }
    return (T *)data;
  }

  inline void *GetPtrWithOffset(size_t offset) const {
    // FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    if (data == nullptr) {
      return (void *)data;
    } else {
      CHECK_WITH_INFO((offset < Size()), "offset is larger than buffer size ");
      return (void *)((char *)data + offset * Tensor::GetTypeSize(type));
    }
  }

  template <typename T>
  inline T *GetPtrWithOffset(size_t offset) const {
    // FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    if (get_tensor_type<T>() != type) {
      TINFER_DEBUG("getVal with type {}, but data type is: {}", GetNumpyTypeDesc(get_tensor_type<T>()),
                   GetNumpyTypeDesc(type));
    }
    if (data == nullptr) {
      return (T *)data;
    } else {
      CHECK_WITH_INFO(offset < Size(), fmt::format("offset ({}) is larger than buffer size ({}) ", offset, Size()));
      return ((T *)data) + offset;
    }
  }

  template <typename T>
  T Max() const {
    if (get_tensor_type<T>() != type) {
      TINFER_DEBUG("getVal with type {}, but data type is: {}", GetNumpyTypeDesc(get_tensor_type<T>()),
                   GetNumpyTypeDesc(type));
    }
    CHECK_WITH_INFO(shape.size() > 0 && data != nullptr, "Should be a non-empty tensor.");
    CHECK_WITH_INFO(where == MEMORY_CPU || where == MEMORY_CPU_PINNED,
                    "max() supports MEMORY_CPU or MEMORY_CPU_PINNED tensor.");
    size_t max_idx = 0;
    T max_val = GetVal<T>(max_idx);
    for (size_t i = 1; i < Size(); ++i) {
      T val = GetVal<T>(i);
      if (val > max_val) {
        max_idx = i;
        max_val = val;
      }
    }
    return max_val;
  }

  template <typename T>
  T Min() const {
    if (get_tensor_type<T>() != type) {
      TINFER_DEBUG("getVal with type {}, but data type is: {}", GetNumpyTypeDesc(get_tensor_type<T>()),
                   GetNumpyTypeDesc(type));
    }
    CHECK_WITH_INFO(shape.size() > 0 && data != nullptr, "Should be a non-empty tensor.");
    CHECK_WITH_INFO(where == MEMORY_CPU || where == MEMORY_CPU_PINNED,
                    "min() supports MEMORY_CPU or MEMORY_CPU_PINNED tensor.");
    size_t min_idx = 0;
    T min_val = GetVal<T>(min_idx);
    for (size_t i = 1; i < Size(); ++i) {
      T val = GetVal<T>(i);
      if (val < min_val) {
        min_idx = i;
        min_val = val;
      }
    }
    return min_val;
  }

  template <typename T>
  T Any(T val) const {
    if (get_tensor_type<T>() != type) {
      TINFER_DEBUG("getVal with type {}, but data type is: {}", GetNumpyTypeDesc(get_tensor_type<T>()),
                   GetNumpyTypeDesc(type));
    }
    CHECK_WITH_INFO(shape.size() > 0 && data != nullptr, "Should be a non-empty tensor.");
    CHECK_WITH_INFO(where == MEMORY_CPU || where == MEMORY_CPU_PINNED,
                    "any() supports MEMORY_CPU or MEMORY_CPU_PINNED tensor.");
    for (size_t i = 0; i < Size(); ++i) {
      if (GetVal<T>(i) == val) {
        return true;
      }
    }
    return false;
  }

  template <typename T>
  T All(T val) const {
    if (get_tensor_type<T>() != type) {
      TINFER_DEBUG("getVal with type %s, but data type is: %s", GetNumpyTypeDesc(get_tensor_type<T>()),
                   GetNumpyTypeDesc(type));
    }
    CHECK_WITH_INFO(shape.size() > 0 && data != nullptr, "Should be a non-empty tensor.");
    CHECK_WITH_INFO(where == MEMORY_CPU || where == MEMORY_CPU_PINNED,
                    "all() supports MEMORY_CPU or MEMORY_CPU_PINNED tensor.");
    for (size_t i = 0; i < Size(); ++i) {
      if (GetVal<T>(i) != val) {
        return false;
      }
    }
    return true;
  }

  void UpdateShape(size_t idx, size_t val) {
    // TODO: find a better way to update the shape
    std::vector<size_t> &shape_ref = const_cast<std::vector<size_t> &>(shape);
    shape_ref[idx] = val;
  }

  template <typename T>
  void Set(CudaAllocator *allocator, std::vector<size_t> data_shape, void *new_data = nullptr) {
    size_t total = 1;
    for (size_t v : data_shape) {
      total *= v;
    }
    void *update_data = allocator->reMalloc((T *)data, total * sizeof(T));
    std::vector<size_t> &shape_ref = const_cast<std::vector<size_t> &>(shape);
    shape_ref = data_shape;
    if (nullptr != new_data) {
      CUDA_CHECK(
          cudaMemcpyAsync(update_data, new_data, sizeof(T) * total, cudaMemcpyDefault, allocator->returnStream()));
    }
    data = update_data;
    const_cast<MemoryType &>(where) = MemoryType::MEMORY_GPU;
    const_cast<DataType &>(type) = get_tensor_type<T>();
  }

  template <typename T>
  void Set(std::vector<size_t> data_shape, void *new_data = nullptr) {
    size_t total = 1;
    for (size_t v : data_shape) {
      total *= v;
    }
    // void *update_data = allocator->reMalloc((T *)data, total * sizeof(T));
    void *update_data = malloc(total * sizeof(T));
    std::vector<size_t> &shape_ref = const_cast<std::vector<size_t> &>(shape);
    shape_ref = data_shape;
    if (nullptr != new_data) {
      memcpy(update_data, new_data, sizeof(T) * total);
    }
    data = update_data;
    cleanup_ = make_cleanup([update_data]() { free(update_data); });
    const_cast<MemoryType &>(where) = MemoryType::MEMORY_CPU;
    const_cast<DataType &>(type) = get_tensor_type<T>();
  }

  Tensor Slice(std::vector<size_t> shape, size_t offset = 0) const;

 private:
  static void ParseNpyIntro(FILE *&f_ptr, uint32_t &header_len, uint32_t &start_data);
  static int ParseNpyHeader(FILE *&f_ptr, uint32_t header_len, DataType &type, std::vector<size_t> &shape);
};

using TensorRawMap = absl::flat_hash_map<std::string, Tensor>;

class TensorMap {
 private:
  TensorRawMap tensor_map_;

  inline bool isValid(const Tensor &tensor) { return tensor.Size() > 0 && tensor.data != nullptr; }

 public:
  TensorMap() = default;
  TensorMap(const TensorRawMap &tensor_map);
  TensorMap(const std::vector<Tensor> &tensor_map);
  TensorMap(std::initializer_list<std::pair<std::string, Tensor>> tensor_map);
  ~TensorMap();

  inline size_t Size() const { return tensor_map_.size(); }

  inline bool IsExist(const std::string &key) const {
    // FT_LOG_DEBUG("%s for key: %s", __PRETTY_FUNCTION__, key.c_str());
    return tensor_map_.find(key) != tensor_map_.end();
  }

  std::vector<std::string> Keys() const;

  inline void Insert(const std::string &key, const Tensor &value) {
    CHECK_WITH_INFO(!IsExist(key), fmt::format("Duplicated key {}", key));
    CHECK_WITH_INFO(isValid(value), fmt::format("A none tensor or nullptr is not allowed (key is {})", key));
    tensor_map_.insert({key, value});
  }

  inline void InsertIfValid(const std::string &key, const Tensor &value) {
    if (isValid(value)) {
      Insert({key, value});
    }
  }

  inline void Insert(std::pair<std::string, Tensor> p) { tensor_map_.insert(p); }

  // prevent converting int or size_t to string automatically
  Tensor At(int tmp) = delete;
  Tensor At(size_t tmp) = delete;

  inline Tensor &At(const std::string &key) {
    // FT_LOG_DEBUG("%s for key %s", __PRETTY_FUNCTION__, key.c_str());
    CHECK_WITH_INFO(IsExist(key), fmt::format("Cannot find a tensor of name {} in the tensor map (keys:{}) ", key,
                                              absl::StrJoin(Keys(), ",")));
    return tensor_map_.at(key);
  }

  inline Tensor At(const std::string &key) const {
    CHECK_WITH_INFO(IsExist(key), fmt::format("Cannot find a tensor of name {} in the tensor map (keys:{})", key,
                                              absl::StrJoin(Keys(), ",")));
    return tensor_map_.at(key);
  }

  inline Tensor &At(const std::string &key, Tensor &default_tensor) {
    //   FT_LOG_DEBUG("%s for key %s", __PRETTY_FUNCTION__, key.c_str());
    if (IsExist(key)) {
      return tensor_map_.at(key);
    }
    return default_tensor;
  }

  inline Tensor At(const std::string &key, Tensor &default_tensor) const {
    //   FT_LOG_DEBUG("%s for key %s", __PRETTY_FUNCTION__, key.c_str());
    if (IsExist(key)) {
      return tensor_map_.at(key);
    }
    return default_tensor;
  }

  inline Tensor &At(const std::string &key, Tensor &&default_tensor) {
    //   FT_LOG_DEBUG("%s for key %s", __PRETTY_FUNCTION__, key.c_str());
    if (IsExist(key)) {
      return tensor_map_.at(key);
    }
    return default_tensor;
  }

  inline Tensor At(const std::string &key, Tensor &&default_tensor) const {
    if (IsExist(key)) {
      return tensor_map_.at(key);
    }
    return default_tensor;
  }

  template <typename T>
  inline T GetVal(const std::string &key) const {
    CHECK_WITH_INFO(IsExist(key), fmt::format("Cannot find a tensor of name {} in the tensor map (keys:{})", key,
                                              absl::StrJoin(Keys(), ",")));
    return tensor_map_.at(key).GetVal<T>();
  }

  template <typename T>
  inline T GetVal(const std::string &key, T default_value) const {
    if (IsExist(key)) {
      return tensor_map_.at(key).GetVal<T>();
    }
    return default_value;
  }

  template <typename T>
  inline T GetValWithOffset(const std::string &key, size_t index) const {
    CHECK_WITH_INFO(IsExist(key), fmt::format("Cannot find a tensor of name {} in the tensor map (keys: {})",
                                              key.c_str(), absl::StrJoin(Keys(), ",")));
    return tensor_map_.at(key).GetVal<T>(index);
  }

  template <typename T>
  inline T GetValWithOffset(const std::string &key, size_t index, T default_value) const {
    if (IsExist(key)) {
      return tensor_map_.at(key).GetVal<T>(index);
    }
    return default_value;
  }

  template <typename T>
  inline T *GetPtr(const std::string &key) const {
    CHECK_WITH_INFO(IsExist(key), fmt::format("Cannot find a tensor of name {} in the tensor map (keys: {})",
                                              key.c_str(), absl::StrJoin(Keys(), ",")));
    return tensor_map_.at(key).GetPtr<T>();
  }

  template <typename T>
  inline T *GetPtr(const std::string &key, T *default_ptr) const {
    if (IsExist(key)) {
      return tensor_map_.at(key).GetPtr<T>();
    }
    return default_ptr;
  }

  template <typename T>
  inline T *GetPtrWithOffset(const std::string &key, size_t index) const {
    CHECK_WITH_INFO(IsExist(key), fmt::format("Cannot find a tensor of name {} in the tensor map (keys: {})",
                                              key.c_str(), absl::StrJoin(Keys(), ",")));
    return tensor_map_.at(key).GetPtrWithOffset<T>(index);
  }

  template <typename T>
  inline T *GetPtrWithOffset(const std::string &key, size_t index, T *default_ptr) const {
    if (IsExist(key)) {
      return tensor_map_.at(key).GetPtrWithOffset<T>(index);
    }
    return default_ptr;
  }

  inline const TensorRawMap &GetMap() const { return tensor_map_; }

  std::string ToString();
  static TensorMap FromNpyFolder(const std::string &base_folder);
  void SaveNpy(const std::string &base_folder);
};

}  // namespace tinfer
