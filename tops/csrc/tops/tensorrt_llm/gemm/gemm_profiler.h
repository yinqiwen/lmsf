/*
** BSD 3-Clause License
**
** Copyright (c) 2023, qiyingwang <qiyingwang@tencent.com>, the respective contributors, as shown by the AUTHORS file.
** All rights reserved.
**
** Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are met:
** * Redistributions of source code must retain the above copyright notice, this
** list of conditions and the following disclaimer.
**
** * Redistributions in binary form must reproduce the above copyright notice,
** this list of conditions and the following disclaimer in the documentation
** and/or other materials provided with the distribution.
**
** * Neither the name of the copyright holder nor the names of its
** contributors may be used to endorse or promote products derived from
** this software without specific prior written permission.
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
** AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
** IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
** DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
** FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
** DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
** SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
** CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
** OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include "tops/tensorrt_llm/common/cublasMMWrapper.h"
#include "tops/tensorrt_llm/common/cudaUtils.h"

#define GEMM_CONFIG "gemm_config.in"

namespace tensorrt_llm {
namespace gemm {
using tensorrt_llm::common::CublasDataType;
struct GemmDims {
  int32_t minM;
  int32_t maxM;
  int32_t n;
  int32_t k;

  GemmDims() : minM(-1), maxM(-1), n(-1), k(-1) {}

  GemmDims(int32_t minM_, int32_t maxM_, int32_t n_, int32_t k_) : minM(minM_), maxM(maxM_), n(n_), k(k_) {}

  bool isInitialized() const { return minM >= 0 && maxM >= 0 && n >= 0 && k >= 0; }
};

// Unique ID of GEMM
// In our case GEMM is uniqly identified by N and K
class GemmIdCore {
 public:
  int n;
  int k;
  CublasDataType dtype;

  GemmIdCore(int n_, int k_, CublasDataType dtype_) : n(n_), k(k_), dtype(dtype_) {}

  GemmIdCore()
      : n(-1),
        k(-1),
        dtype(CublasDataType::FLOAT_DATATYPE)  // dtype does not matter here
  {}
  void serialize(char*& buffer) const;
  void deserialize(const char*& data);
  bool operator==(const GemmIdCore& id) const { return isEqual(id); }

  friend std::ostream& operator<<(std::ostream& out, const GemmIdCore& id) {
    out << "(N;K)=(" << id.n << ";" << id.k << "),";
    out << " type=" << static_cast<int>(id.dtype);
    return out;
  }

 protected:
  bool isEqual(const GemmIdCore& id) const { return n == id.n && k == id.k && dtype == id.dtype; }
};

// Hash of GemmId
struct GemmIdCoreHash {
  std::size_t operator()(const GemmIdCore& id) const {
    auto h1 = std::hash<int>{}(id.n);
    auto h2 = std::hash<int>{}(id.k);
    auto h3 = std::hash<int>{}(static_cast<int>(id.dtype));
    return h1 ^ h2 ^ h3;
  }
};

class GemmIdCublas : public GemmIdCore {
 public:
  bool transA{};
  bool transB{};

  GemmIdCublas(int n_, int k_, CublasDataType dtype_, bool transA_, bool transB_)
      : GemmIdCore(n_, k_, dtype_), transA(transA_), transB(transB_) {}

  GemmIdCublas() {}

  static int getSerializationSize() {
    return sizeof(int) + sizeof(int) + sizeof(CublasDataType) + sizeof(bool) + sizeof(bool);
  }

  void serialize(char*& buffer) const;
  void deserialize(const char*& data);

  bool operator==(const GemmIdCublas& id) const { return isEqual(id) && transA == id.transA && transB == id.transB; }

  friend std::ostream& operator<<(std::ostream& out, const GemmIdCublas& id) {
    out << "(N;K)=(" << id.n << ";" << id.k << "),";
    out << " type=" << static_cast<int>(id.dtype);
    out << " transA=" << id.transA;
    out << " transB=" << id.transB;
    return out;
  }
};

// Hash of GemmIdCublas
struct GemmIdCublasHash {
  std::size_t operator()(const GemmIdCublas& id) const {
    auto h1 = std::hash<int>{}(id.n);
    auto h2 = std::hash<int>{}(id.k);
    auto h3 = std::hash<int>{}(static_cast<int>(id.dtype));
    auto h4 = std::hash<bool>{}(id.transA);
    auto h5 = std::hash<bool>{}(id.transB);
    return h1 ^ h2 ^ h3 ^ h4 ^ h5;
  }
};

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
class GemmPluginProfiler {
 public:
  static constexpr int MAX_PROFILE_M = 8192;

  // Map for single GEMM for different Ms (GEMM dimension) to the best config for particular M
  using MProfileMap = std::unordered_map<int, std::optional<Config>>;
  using MProfileMapPtr = std::shared_ptr<MProfileMap>;

  // requires exclusive ownership to write to *this
  using reader_lock = std::unique_lock<std::shared_timed_mutex>;
  // requires shared ownership to read from other
  using writer_lock = std::shared_lock<std::shared_timed_mutex>;

  // Struct of continuing map if GEMMs to the best profiles for different Ms
  struct MNKProfileMap {
    // Mutex guarding map
    std::shared_timed_mutex mutex;
    // Map from GEMM Id to profile for particular GEMM
    std::unordered_map<GemmIdType, MProfileMapPtr, GemmIdHashType> profileMap;

    bool existsMProfileMap(const GemmIdType& id) {
      const auto iter = profileMap.find(id);
      return iter != profileMap.end();
    }

    void createMProfileMap(const GemmIdType& id) { profileMap[id] = std::make_shared<MProfileMap>(); }

    MProfileMapPtr getMProfileMap(const GemmIdType& id) {
      const auto iter = profileMap.find(id);
      if (iter == profileMap.end()) {
        std::ostringstream msg;
        msg << "Cannot find ID (" << id << ") in the profile map. Abort.";
        TLLM_LOG_ERROR(msg.str());
      }
      return iter->second;
    }
  };

  using MNKProfileMapPtr = std::shared_ptr<MNKProfileMap>;

  GemmPluginProfiler();

  //   // Write values into buffer
  //   template <typename T>
  //   void write(char*& buffer, const T& val) {
  //     std::memcpy(buffer, &val, sizeof(T));
  //     buffer += sizeof(T);
  //   }

  //   // Read values from buffer
  //   template <typename T>
  //   void read(const char*& buffer, T& val) {
  //     std::memcpy(&val, buffer, sizeof(T));
  //     buffer += sizeof(T);
  //   }
  int save(const std::string& file) const;
  int load(const std::string& file);

  void serialize(char*& buffer, const GemmIdType& gemmId) const;

  void deserialize(const char*& data, const GemmIdType& gemmId);
  size_t getSerializationSize(const GemmIdType& gemmId) const;

  void profileTactics(const RunnerPtr& runner, CublasDataType type, const GemmDims& dims, const GemmIdType& gemmId);

  void setSelectionTactics(const MNKProfileMapPtr& map) { mMNKProfileMap = map; }

  void setTmpWorkspaceSizeInBytes(size_t bytes) { mTmpWorkspaceSizeInBytes = bytes; }

  void setSkip(bool skip) { mSkip = mSkip || skip; }

  std::optional<Config> getBestConfig(int m, const GemmIdType& gemmId) const;

 protected:
  virtual void runTactic(int m, int n, int k, const Config& tactic, char* workspace, const cudaStream_t& stream) = 0;

  virtual void computeTmpSize(int maxM, int n, int k) = 0;

  virtual bool checkTactic(int m, int n, int k, const Config& tactic) const { return true; }

  virtual std::vector<Config> getTactics(int m, int n, int k) const = 0;

  virtual void initTmpData(int m, int n, int k, char* workspace, size_t size, cudaStream_t stream){};

 private:
  void allocateTmpData();

  void freeTmpData();

  std::optional<Config> profileTacticsForProblem(int m, int n, int k, const std::vector<Config>& tactics);

  float profileTacticForProblem(int m, int n, int k, const Config& tactic);

  int nextPowerOfTwo(int v) const {
    --v;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return ++v;
  }

 protected:
  RunnerPtr mRunner{nullptr};

  CublasDataType mType{};

 private:
  MNKProfileMapPtr mMNKProfileMap{};

  size_t mTmpWorkspaceSizeInBytes{0};

  char* mWorkspaceTmp{nullptr};

  // GemmDims mDims{};

  bool mSkip{false};
};

template <typename GemmPluginProfilerType>
class GemmPluginProfilerManager {
 public:
  using MNKProfileMap = typename GemmPluginProfilerType::MNKProfileMap;
  using MNKProfileMapPtr = typename GemmPluginProfilerType::MNKProfileMapPtr;
  using GemmPluginProfilerPtr = std::shared_ptr<GemmPluginProfilerType>;

  GemmPluginProfilerManager() { mMNKProfileMap = std::make_shared<MNKProfileMap>(); }

  GemmPluginProfilerPtr createGemmPluginProfiler(bool inference, bool skip = false) {
    auto profiler = std::make_shared<GemmPluginProfilerType>();
    profiler->setSkip(skip);
    // If the profiler is created during the engine build,
    // mMNKProfileMap is shared between different profilers to minimize the time spent on the profiling
    // and do not repeat profiling for the GEMMs of the same shape.
    if (!inference) {
      profiler->setSelectionTactics(mMNKProfileMap);
    }
    return profiler;
  }

 private:
  MNKProfileMapPtr mMNKProfileMap{};
};

using CublasGemmWrapper = tensorrt_llm::common::CublasMMWrapper;
using CublasGemmWrapperPtr = std::shared_ptr<CublasGemmWrapper>;

class CublasLtGemmPluginProfiler
    : public GemmPluginProfiler<cublasLtMatmulHeuristicResult_t, CublasGemmWrapperPtr, GemmIdCublas, GemmIdCublasHash> {
 public:
  using Config = cublasLtMatmulHeuristicResult_t;

  static void getProblemParams(cublasOperation_t& transa, cublasOperation_t& transb, int& m, int& n, int& k, int& lda,
                               int& ldb, int& ldc, int transA, int transB, int M, int N, int K);
  static int32_t computeMDimension(int transA, const int32_t nbDims, const int64_t* dims);
  static int32_t computeNDimension(int transB, const int32_t nbDims, const int64_t* dims);
  static void runGemm(const int M, const int N, const int K, const bool transA, const bool transB,
                      const CublasGemmWrapperPtr& cublasWrapperPtr, const void* act, const void* weight, void* output,
                      const std::optional<cublasLtMatmulHeuristicResult_t>& heuristic, void* workspace,
                      cudaStream_t stream);

  void setTranspose(bool transposeA, bool transposeB) {
    mTransA = transposeA;
    mTransB = transposeB;
  }

  void setOutputType(CublasDataType type) { mOutputType = type; }

 protected:
  void runTactic(int m, int n, int k, const Config& tactic, char* workspace, const cudaStream_t& stream) override;

  void computeTmpSize(int maxM, int n, int k) override;

  bool checkTactic(int m, int n, int k, const Config& tactic) const override;

  std::vector<Config> getTactics(int m, int n, int k) const override;

 private:
  bool mTransA;
  bool mTransB;
  CublasDataType mOutputType;

  static constexpr size_t ALIGNMENT = 256;
};

}  // namespace gemm
}  // namespace tensorrt_llm