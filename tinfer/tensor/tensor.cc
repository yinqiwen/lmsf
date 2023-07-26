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

#include "tinfer/tensor/tensor.h"
#include <cuda_runtime_api.h>
#include <sstream>

namespace tinfer {

Tensor::Tensor()
    :  // a none tensor.
      where(MEMORY_CPU),
      type(TYPE_INVALID),
      shape({}),
      data(nullptr),
      offsets({})  // only a record to record offset
{}

Tensor::Tensor(const MemoryType _where, const DataType _type, const std::vector<size_t> _shape, const void *_data)
    : where(_where), type(_type), shape(_shape), data(_data) {}

Tensor::Tensor(const MemoryType _where, const DataType _type, const std::vector<size_t> _shape, const void *_data,
               const std::vector<size_t> _offset)
    : where(_where), type(_type), shape(_shape), data(_data), offsets(_offset) {}

void Tensor::ParseNpyIntro(FILE *&f_ptr, uint32_t &header_len, uint32_t &start_data) {
  const char magic[] =
      "\x93"
      "NUMPY";
  char magic_test[sizeof(magic)] = "\0";

  size_t n_elems = fread((void *)magic_test, sizeof(char), sizeof(magic) - 1, f_ptr);
  if (n_elems != sizeof(magic) - 1 || std::string(magic) != std::string(magic_test)) {
    throw std::runtime_error("Could read magic token in NPY file");
  }

  uint8_t npy_major = 0;
  uint8_t npy_minor = 0;
  n_elems = fread((void *)&npy_major, sizeof(uint8_t), 1, f_ptr);
  n_elems += fread((void *)&npy_minor, sizeof(uint8_t), 1, f_ptr);

  if (npy_major == 1) {
    uint16_t header_len_u16 = 0;
    n_elems = fread((void *)&header_len_u16, sizeof(uint16_t), 1, f_ptr);
    header_len = header_len_u16;
  } else if (npy_major == 2) {
    uint32_t header_len_u32 = 0;
    n_elems = fread((void *)&header_len_u32, sizeof(uint32_t), 1, f_ptr);
    header_len = header_len_u32;
  } else {
    throw std::runtime_error("Unsupported npy version: " + std::to_string(npy_major));
  }

  start_data = 8 + 2 * npy_major + header_len;
}

int Tensor::ParseNpyHeader(FILE *&f_ptr, uint32_t header_len, DataType &type, std::vector<size_t> &shape) {
  char *header_c = (char *)malloc(header_len * sizeof(char));
  size_t n_elems = fread((void *)header_c, sizeof(char), header_len, f_ptr);
  if (n_elems != header_len) {
    free(header_c);
    return -1;
  }
  std::string header(header_c, header_len);
  free(header_c);

  size_t start, end;
  start = header.find("'descr'") + 7;
  start = header.find("'", start);
  end = header.find("'", start + 1);
  type = TypeFromNumpyDesc(header.substr(start + 1, end - start - 1));

  start = header.find("'fortran_order'") + 15;
  start = header.find(":", start);
  end = header.find(",", start + 1);
  if (header.substr(start + 1, end - start - 1).find("False") == std::string::npos) {
    throw std::runtime_error("Unsupported value for fortran_order while reading npy file");
  }

  start = header.find("'shape'") + 7;
  start = header.find("(", start);
  end = header.find(")", start + 1);

  std::istringstream shape_stream(header.substr(start + 1, end - start - 1));
  std::string token;

  shape.clear();
  while (std::getline(shape_stream, token, ',')) {
    if (token.find_first_not_of(' ') == std::string::npos) {
      break;
    }
    shape.push_back(std::stoul(token));
  }

  return 0;
}

Tensor Tensor::LoadNpy(const std::string &npy_file, const MemoryType where) {
  DataType type;
  std::vector<size_t> shape;

  FILE *f_ptr = fopen(npy_file.c_str(), "rb");
  if (f_ptr == nullptr) {
    throw std::runtime_error("Could not open file " + npy_file);
  }
  uint32_t header_len, start_data;
  ParseNpyIntro(f_ptr, header_len, start_data);
  ParseNpyHeader(f_ptr, header_len, type, shape);

  const size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
  void *data_cpu = malloc(size * Tensor::GetTypeSize(type));
  void *data = data_cpu;

  size_t n_elems = fread(data_cpu, Tensor::GetTypeSize(type), size, f_ptr);
  CHECK_WITH_INFO(n_elems == size, "reading tensor failed");
  if (where == MEMORY_GPU) {
    cudaMalloc(&data, size * Tensor::GetTypeSize(type));
    cudaMemcpy(data, data_cpu, size * Tensor::GetTypeSize(type), cudaMemcpyHostToDevice);
    free(data_cpu);
  }

  fclose(f_ptr);
  return Tensor(where, type, shape, data);
}

size_t Tensor::Size() const {
  if (data == nullptr || shape.size() == 0) {
    return 0;
  }
  return std::accumulate(shape.begin(), shape.end(), (size_t)1, std::multiplies<size_t>());
}

size_t Tensor::SizeBytes() const { return Size() * Tensor::GetTypeSize(type); }

std::string Tensor::WhereToString() const {
  static const std::unordered_map<MemoryType, std::string> mem_to_string{
      {MEMORY_CPU, "CPU"}, {MEMORY_CPU_PINNED, "CPU_PINNED"}, {MEMORY_GPU, "GPU"}};
  return mem_to_string.at(where);
}

std::string Tensor::ToString() const {
  std::string memtype_str = WhereToString();

  static const std::unordered_map<DataType, std::string> type_to_string{
      {TYPE_BOOL, "BOOL"},     {TYPE_UINT8, "UINT8"}, {TYPE_UINT16, "UINT16"},   {TYPE_UINT32, "UINT32"},
      {TYPE_UINT64, "UINT64"}, {TYPE_INT8, "INT8"},   {TYPE_INT16, "INT16"},     {TYPE_INT32, "INT32"},
      {TYPE_INT64, "INT64"},   {TYPE_BF16, "BF16"},   {TYPE_FP16, "FP16"},       {TYPE_FP32, "FP32"},
      {TYPE_FP64, "FP64"},     {TYPE_BYTES, "BYTES"}, {TYPE_INVALID, "INVALID"}, {TYPE_FP8_E4M3, "E4M3"},
      {TYPE_VOID, "VOID"},
  };
  return fmt::format("Tensor[where={}, type={}, shape=[{}], data={}]", memtype_str.c_str(),
                     type_to_string.at(type).c_str(), absl::StrJoin(shape, ","), data);
}

DataType Tensor::TypeFromNumpyDesc(std::string type) {
  static const std::unordered_map<std::string, DataType> type_map{
      {"?", TYPE_BOOL},    {"b", TYPE_BYTES}, {"u1", TYPE_UINT8}, {"u2", TYPE_UINT16}, {"u4", TYPE_UINT32},
      {"u8", TYPE_UINT64}, {"i1", TYPE_INT8}, {"i2", TYPE_INT16}, {"i4", TYPE_INT32},  {"i8", TYPE_INT64},
      {"f2", TYPE_FP16},   {"f4", TYPE_FP32}, {"f8", TYPE_FP64}};
  return type_map.at(type);
}

size_t Tensor::GetTypeSize(DataType type) {
  static const std::unordered_map<DataType, size_t> type_map{{TYPE_BOOL, sizeof(bool)},
                                                             {TYPE_BYTES, sizeof(char)},
                                                             {TYPE_UINT8, sizeof(uint8_t)},
                                                             {TYPE_UINT16, sizeof(uint16_t)},
                                                             {TYPE_UINT32, sizeof(uint32_t)},
                                                             {TYPE_UINT64, sizeof(uint64_t)},
                                                             {TYPE_INT8, sizeof(int8_t)},
                                                             {TYPE_INT16, sizeof(int16_t)},
                                                             {TYPE_INT32, sizeof(int32_t)},
                                                             {TYPE_INT64, sizeof(int64_t)},
#ifdef ENABLE_BF16
                                                             {TYPE_BF16, sizeof(__nv_bfloat16)},
#endif
#ifdef ENABLE_FP8
                                                             {TYPE_FP8_E4M3, sizeof(__nv_fp8_e4m3)},
#endif
                                                             {TYPE_FP16, sizeof(half)},
                                                             {TYPE_FP32, sizeof(float)},
                                                             {TYPE_FP64, sizeof(double)}};
  return type_map.at(type);
}

std::string Tensor::GetNumpyTypeDesc(DataType type) const {
  static const std::unordered_map<DataType, std::string> type_map{
      {TYPE_INVALID, "x"}, {TYPE_BOOL, "?"},    {TYPE_BYTES, "b"}, {TYPE_UINT8, "u1"}, {TYPE_UINT16, "u2"},
      {TYPE_UINT32, "u4"}, {TYPE_UINT64, "u8"}, {TYPE_INT8, "i1"}, {TYPE_INT16, "i2"}, {TYPE_INT32, "i4"},
      {TYPE_INT64, "i8"},  {TYPE_FP16, "f2"},   {TYPE_FP32, "f4"}, {TYPE_FP64, "f8"}};

  if (type == TYPE_BF16) {
    TINFER_WARN(
        "getNumpyTypeDesc(TYPE_BF16) returns an invalid type 'x' since "
        "Numpy doesn't support bfloat16 as of now, it will be properly "
        "extended if numpy supports.Please refer for the discussions "
        "https://github.com/numpy/numpy/issues/19808.");
  }

  return type_map.count(type) > 0 ? type_map.at(type) : "x";
}

void Tensor::SaveNpy(const std::string &filename) const {
  // Save tensor to NPY 1.0 format (see
  // https://numpy.org/neps/nep-0001-npy-format.html)
  void *cpu_data = (void *)data;
  bool is_data_temp = false;
  size_t tensor_size = Size();
  if (where == MemoryType::MEMORY_GPU) {
    cpu_data = malloc(tensor_size * Tensor::GetTypeSize(type));
    is_data_temp = true;
    cudaDeviceSynchronize();
    cudaMemcpy(cpu_data, data, tensor_size * Tensor::GetTypeSize(type), cudaMemcpyDeviceToHost);
  }

  const char magic[] =
      "\x93"
      "NUMPY";
  const uint8_t npy_major = 1;
  const uint8_t npy_minor = 0;

  std::stringstream header_stream;
  header_stream << "{'descr': '" << GetNumpyTypeDesc(type) << "', 'fortran_order': False, 'shape': (";
  for (size_t i = 0; i < shape.size(); ++i) {
    header_stream << shape[i];
    if (i + 1 < shape.size() || shape.size() == 1) {
      header_stream << ", ";
    }
  }
  header_stream << ")}";
  int base_length = 6 + 4 + header_stream.str().size();
  int pad_length = 16 * ((base_length + 1 + 15) / 16);  // Take ceiling of base_length + 1 (for '\n' ending)
  for (int i = 0; i < pad_length - base_length; ++i) {
    header_stream << ((i == pad_length - base_length - 1) ? "\n" : "\x20");
  }
  std::string header = header_stream.str();
  const uint16_t header_len = header.size();

  FILE *f_ptr = fopen(filename.c_str(), "wb");
  CHECK_WITH_INFO(f_ptr != nullptr, fmt::format("Unable to open {} for writing", filename));

  fwrite(magic, sizeof(char), sizeof(magic) - 1, f_ptr);
  fwrite(&npy_major, sizeof(uint8_t), 1, f_ptr);
  fwrite(&npy_minor, sizeof(uint8_t), 1, f_ptr);
  fwrite(&header_len, sizeof(uint16_t), 1, f_ptr);
  fwrite(header.c_str(), sizeof(char), header_len, f_ptr);
  fwrite(cpu_data, Tensor::GetTypeSize(type), tensor_size, f_ptr);

  fclose(f_ptr);

  if (is_data_temp) {
    free(cpu_data);
  }
}

Tensor Tensor::Slice(std::vector<size_t> shape, size_t offset) const {
  if (this->data != nullptr) {
    size_t n_elts = this->Size();
    size_t n_sliced_elts = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    CHECK_WITH_INFO(n_sliced_elts + offset <= n_elts,
                    fmt::format("The number ({}) of elements of sliced tensor exceeds "
                                "that({}) of the original tensor ",
                                n_sliced_elts + offset, n_elts));
  }
  return Tensor(this->where, this->type, shape, this->GetPtrWithOffset(offset));
}

TensorMap::TensorMap(const TensorRawMap &tensor_map) {
  for (auto &kv : tensor_map) {
    if (isValid(kv.second)) {
      Insert(kv.first, kv.second);
    } else {
      TINFER_DEBUG("{} is not a valid tensor, skipping insertinto TensorMap", kv.first);
    }
  }
}

TensorMap::TensorMap(const std::vector<Tensor> &tensor_map) {
  for (size_t i = 0; i < tensor_map.size(); i++) {
    Insert(std::to_string(i), tensor_map[i]);
  }
}

TensorMap::TensorMap(std::initializer_list<std::pair<std::string, Tensor>> tensor_map) {
  for (auto &pair : tensor_map) {
    if (isValid(pair.second)) {
      Insert(pair.first, pair.second);
    } else {
      TINFER_DEBUG("{} is not a valid tensor, skipping insert into TensorMap", pair.first);
    }
  }
}

TensorMap::~TensorMap() { tensor_map_.clear(); }

std::vector<std::string> TensorMap::Keys() const {
  std::vector<std::string> key_names;
  for (auto &kv : tensor_map_) {
    key_names.push_back(kv.first);
  }
  return key_names;
}

std::string TensorMap::ToString() {
  std::stringstream ss;
  ss << "{";
  std::vector<std::string> key_names = Keys();
  for (size_t i = 0; i < tensor_map_.size(); ++i) {
    ss << key_names[i] << ": " << At(key_names[i]).ToString();
    if (i < tensor_map_.size() - 1) {
      ss << ", ";
    }
  }
  ss << "}";
  return ss.str();
}

TensorMap TensorMap::FromNpyFolder(const std::string &base_folder) {
  DIR *dir_p = opendir(base_folder.c_str());
  CHECK_WITH_INFO(dir_p != nullptr, fmt::format("Could not open folder {}. ", base_folder));
  struct dirent *dp;

  TensorMap ret_tensor;
  while ((dp = readdir(dir_p)) != nullptr) {
    std::string filename(dp->d_name);
    size_t len = filename.length();
    if (len < 4 || filename.compare(len - 4, 4, ".npy")) {
      continue;
    }

    size_t pos = filename.find('-');
    CHECK_WITH_INFO(pos != std::string::npos, fmt::format("Invalid filename: {}", filename));

    MemoryType where = MEMORY_CPU;
    if (filename.compare(0, pos, "GPU") == 0) {
      where = MEMORY_GPU;
    } else if (filename.compare(0, pos, "CPU") == 0) {
      where = MEMORY_CPU;
    } else if (filename.compare(0, pos, "CPU_PINNED") == 0) {
      where = MEMORY_CPU_PINNED;
    } else {
      CHECK_WITH_INFO(false, fmt::format("Invalid filename: {}", filename));
    }
    std::string key = filename.substr(pos + 1, len - pos - 5);

    ret_tensor.tensor_map_.insert({key, Tensor::LoadNpy(base_folder + "/" + filename, where)});
  }

  closedir(dir_p);

  return ret_tensor;
}

void TensorMap::SaveNpy(const std::string &base_folder) {
  mode_t mode_0755 = S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH;
  int ret = mkdir(base_folder.c_str(), mode_0755);
  CHECK_WITH_INFO(ret == 0 || errno == EEXIST, fmt::format("Could not create folder {}", base_folder));

  for (const auto &item : tensor_map_) {
    item.second.SaveNpy(base_folder + "/" + item.second.WhereToString() + "-" + item.first + ".npy");
  }
}

}  // namespace tinfer
