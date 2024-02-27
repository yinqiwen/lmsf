/*
** BSD 3-Clause License
**
** Copyright (c) 2023, qiyingwang <qiyingwang@tencent.com>, the respective
*contributors, as shown by the AUTHORS file.
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
** IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*ARE
** DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
** FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
** DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
** SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
** CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
** OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include "tops/tensorrt_llm/gemm/gemm_profiler.h"
#include <cstdio>
#include <cstdlib>
#include "tops/tensorrt_llm/common/workspace.h"

// #include
// "tops/tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
// #include "tops/tensorrt_llm/kernels/cutlass_kernels/int8_gemm/int8_gemm.h"

namespace tensorrt_llm {
namespace gemm {

using namespace tensorrt_llm::common;
// Write values into buffer
template <typename T>
void write(char *&buffer, const T &val) {
  std::memcpy(buffer, &val, sizeof(T));
  buffer += sizeof(T);
}

// Read values from buffer
template <typename T>
void read(const char *&buffer, T &val) {
  std::memcpy(&val, buffer, sizeof(T));
  buffer += sizeof(T);
}

void GemmIdCore::serialize(char *&buffer) const {
  write(buffer, n);
  write(buffer, k);
  write(buffer, dtype);
}
void GemmIdCore::deserialize(const char *&data) {
  read(data, n);
  read(data, k);
  read(data, dtype);
}

void GemmIdCublas::serialize(char *&buffer) const {
  GemmIdCore::serialize(buffer);
  write(buffer, transA);
  write(buffer, transB);
}
void GemmIdCublas::deserialize(const char *&data) {
  GemmIdCore::deserialize(data);
  read(data, transA);
  read(data, transB);
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::GemmPluginProfiler() {
  mMNKProfileMap = std::make_shared<MNKProfileMap>();

  // set SKIP_GEMM_PLUGIN_PROFILINGS=1 to avoid tactics profilings
  const auto skipEnv = std::getenv("SKIP_GEMM_PLUGIN_PROFILINGS");
  mSkip = (skipEnv != NULL && std::stoi(skipEnv));
  if (mSkip) {
    TLLM_LOG_DEBUG(
        "SKIP_GEMM_PLUGIN_PROFILINGS is set. Skipping GEMM plugin "
        "profilings. It could result in runtime error "
        "if default tactic is not defined.");
  }
}
template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
int GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::save(const std::string &file) const {
  FILE *fp = fopen(file.c_str(), "w");
  if (fp == NULL) {
    TLLM_LOG_ERROR("Failed to open file to save gemm profiler");
    return -1;
  }

  int n = mMNKProfileMap->profileMap.size();
  int save_buffer_size = sizeof(int);
  for (const auto &[key, val_map] : mMNKProfileMap->profileMap) {
    save_buffer_size += GemmIdType::getSerializationSize();  // key size
    save_buffer_size += getSerializationSize(key);
  }

  char *buffer = new char[save_buffer_size];
  char *orig_buffer = buffer;

  write(buffer, n);
  for (const auto &[key, val_map] : mMNKProfileMap->profileMap) {
    key.serialize(buffer);
    serialize(buffer, key);
  }
  int rc = 0;
  if (buffer - orig_buffer != save_buffer_size) {
    TLLM_LOG_ERROR("Expected to write buffer %dbytes, but %dbytes buffer used!", save_buffer_size,
                   buffer - orig_buffer);
    rc = -1;
  } else {
    size_t write_n = fwrite(orig_buffer, 1, save_buffer_size, fp);

    if (write_n != save_buffer_size) {
      TLLM_LOG_ERROR("Expected to write %dbytes, only %dbytes writed!", save_buffer_size, write_n);
      rc = -1;
    }
  }

  delete[] orig_buffer;
  fclose(fp);
  return rc;
}
template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
int GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::load(const std::string &file) {
  FILE *fp = fopen(file.c_str(), "rb");
  if (fp == NULL) {
    TLLM_LOG_ERROR("Failed to open file:%s to read gemm profiler, use default config.", file.c_str());
    return -1;
  }
  fseek(fp, 0, SEEK_END);
  long fsize = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  char *buffer = new char[fsize];
  size_t read_n = fread(buffer, 1, fsize, fp);
  const char *read_buffer = buffer;
  int rc = 0;
  if (read_n != fsize) {
    TLLM_LOG_ERROR("Expected to read %dbytes, only %dbytes writed!", fsize, read_n);
    rc = -1;
  } else {
    mMNKProfileMap = std::make_shared<MNKProfileMap>();
    int n = 0;
    read(read_buffer, n);

    for (int i = 0; i < n; i++) {
      GemmIdType key;
      key.deserialize(read_buffer);
      deserialize(read_buffer, key);
    }
    if (read_buffer - buffer != fsize) {
      TLLM_LOG_ERROR("Expected to read buffer %dbytes, but %dbytes buffer used!", fsize, read_buffer - buffer);
      rc = -1;
    }
  }
  // printf("####n:%d\n", mMNKProfileMap->profileMap.size());
  delete[] buffer;
  fclose(fp);
  TLLM_LOG_INFO("Load GemmPluginProfiler with %d entries", mMNKProfileMap->profileMap.size());
  return rc;
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
void GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::serialize(char *&buffer,
                                                                                  const GemmIdType &gemmId) const {
  auto mProfileMap = mMNKProfileMap->getMProfileMap(gemmId);

  // Save number of profiles for given GEMM ID
  write(buffer, static_cast<int>(mProfileMap->size()));
  for (const auto &pair : *mProfileMap) {
    // Save pair of M to the best GEMM config
    write(buffer, pair);
  }
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
void GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::deserialize(const char *&data,
                                                                                    const GemmIdType &gemmId) {
  // NOTE: this mutex is not needed since each thread owns its private map, but
  // will put here for consistency
  writer_lock lock(mMNKProfileMap->mutex);

  // GemmId gemmId(dims.n, dims.k);
  if (!mMNKProfileMap->existsMProfileMap(gemmId)) {
    // Create GEMM with GEMM ID if it does not exist
    mMNKProfileMap->createMProfileMap(gemmId);
  }
  // Populate map with profiles of GEMM ID
  auto profileMap = mMNKProfileMap->getMProfileMap(gemmId);
  int selectedMapSize;
  read(data, selectedMapSize);
  for (int ii = 0; ii < selectedMapSize; ++ii) {
    std::pair<int, std::optional<Config>> config;
    read(data, config);
    profileMap->insert(config);
  }
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
size_t GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::getSerializationSize(
    const GemmIdType &gemmId) const {
  reader_lock lock(mMNKProfileMap->mutex);
  return sizeof(int) +  // size of the tactics map
         mMNKProfileMap->getMProfileMap(gemmId)->size() * sizeof(std::pair<int, std::optional<Config>>);  // size of the
                                                                                                          // tactics map
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
void GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::profileTactics(const RunnerPtr &runner,
                                                                                       CublasDataType type,
                                                                                       const GemmDims &dims,
                                                                                       const GemmIdType &gemmId) {
  writer_lock lock(mMNKProfileMap->mutex);

  if (!dims.isInitialized()) {
    return;
  }

  mRunner = runner;
  mType = type;

  const int maxM = std::min(nextPowerOfTwo(dims.maxM), MAX_PROFILE_M);
  computeTmpSize(maxM, dims.n, dims.k);

  if (!mMNKProfileMap->existsMProfileMap(gemmId)) {
    // Create map for GEMM ID
    mMNKProfileMap->createMProfileMap(gemmId);
  }

  if (mSkip) {
    return;
  }

  auto mProfileMap = mMNKProfileMap->getMProfileMap(gemmId);

  auto profileTactics = [&mProfileMap, this](int m, int n, int k) {
    if (mProfileMap->count(m) == 0) {
      initTmpData(m, n, k, mWorkspaceTmp, mTmpWorkspaceSizeInBytes, cudaStreamDefault);
      const auto tactics = this->getTactics(m, n, k);
      // Profile different tactics for particular m and insert best config to
      // the map
      mProfileMap->insert({m, this->profileTacticsForProblem(m, n, k, tactics)});
    }
  };

  // Allocate tmp data to run GEMMs
  allocateTmpData();

  const int startMinMRounded = nextPowerOfTwo(dims.minM);
  for (int m = startMinMRounded; m < maxM; m *= 2) {
    profileTactics(m, dims.n, dims.k);
  }

  profileTactics(maxM, dims.n, dims.k);
  // Free tmp data
  freeTmpData();
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
std::optional<Config> GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::getBestConfig(
    int m, const GemmIdType &gemmId) const {
  reader_lock lock(mMNKProfileMap->mutex);

  if (mSkip) {
    return std::nullopt;
  }
  if (!mMNKProfileMap->existsMProfileMap(gemmId)) {
    return std::nullopt;
  }
  const int mRounded = std::min(nextPowerOfTwo(m), MAX_PROFILE_M);
  auto map = mMNKProfileMap->getMProfileMap(gemmId);
  auto found = map->find(mRounded);
  if (found == map->end()) {
    return std::nullopt;
  }
  return found->second;
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
void GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::allocateTmpData() {
  // TLLM_CHECK_WITH_INFO(mTmpWorkspaceSizeInBytes > 0, "tmpWorkspaceSizeInBytes
  // must be larger than 0");
  const auto status = cudaMalloc(&mWorkspaceTmp, mTmpWorkspaceSizeInBytes);
  // TLLM_CHECK_WITH_INFO(status == cudaSuccess, "Can't allocate tmp workspace
  // for GEMM tactics profiling.");
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
void GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::freeTmpData() {
  const auto status = cudaFree(mWorkspaceTmp);
  // TLLM_CHECK_WITH_INFO(status == cudaSuccess, "Can't free tmp workspace for
  // GEMM tactics profiling.");
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
std::optional<Config> GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::profileTacticsForProblem(
    int m, int n, int k, const std::vector<Config> &tactics) {
  TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);

  // printf("Candidata %d for m:%d, n:%d, k:%d\n", tactics.size(), m, n, k);

  float bestTime = std::numeric_limits<float>::max();
  Config bestConfig;
  bool foundOne = false;

  // Iterate over all tactics for given M, N and K
  for (int ii = 0; ii < tactics.size(); ++ii) {
    const Config &candidateConfig = tactics[ii];
    float time = std::numeric_limits<float>::max();
    try {
      if (!checkTactic(m, n, k, candidateConfig)) {
        continue;
      }
      // Profile particualar tactic for given M, N and K
      time = profileTacticForProblem(m, n, k, candidateConfig);
      foundOne = true;
    } catch (const std::exception &e) {
      std::ostringstream msg;
      msg << "Cannot profile configuration " << ii << " (for"
          << " m=" << m << ", n=" << n << ", k=" << k << ")"
          << ", reason: \"" << e.what() << "\". Skipped";
      TLLM_LOG_WARNING(msg.str());
      continue;
    }

    // Choose the fastest tactic
    if (time < bestTime) {
      bestConfig = candidateConfig;
      bestTime = time;
    }
  }

  if (!foundOne) {
    std::ostringstream msg;
    msg << "Have not found any valid GEMM config for shape ("
        << "m=" << m << ", n=" << n << ", k=" << k << "). Will try to use default or fail at runtime";
    TLLM_LOG_WARNING(msg.str());
    return std::nullopt;
  }
  printf("Fastest time:%fms for m:%d, n:%d, k:%d\n", bestTime, m, n, k);
  return {bestConfig};
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
float GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::profileTacticForProblem(int m, int n, int k,
                                                                                                 const Config &tactic) {
  constexpr int warmup = 5;
  constexpr int runs = 10;

  cudaStream_t stream = cudaStreamDefault;
  // Warmup the execution
  for (int i = 0; i < warmup; ++i) {
    runTactic(m, n, k, tactic, mWorkspaceTmp, stream);
  }

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaDeviceSynchronize();
  cudaEventRecord(start, 0);

  // Profile GEMM
  for (int i = 0; i < runs; ++i) {
    runTactic(m, n, k, tactic, mWorkspaceTmp, stream);
  }

  cudaEventRecord(stop, 0);

  cudaEventSynchronize(stop);

  float elapsed;
  cudaEventElapsedTime(&elapsed, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return elapsed / runs;
}

// template class GemmPluginProfiler<
//     tensorrt_llm::cutlass_extensions::CutlassGemmConfig,
//     std::shared_ptr<tensorrt_llm::kernels::cutlass_kernels::CutlassInt8GemmRunnerInterface>,
//     GemmIdCore, GemmIdCoreHash>;

// template class GemmPluginProfiler<
//     tensorrt_llm::cutlass_extensions::CutlassGemmConfig,
//     std::shared_ptr<tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunnerInterface>,
//     GemmIdCore, GemmIdCoreHash>;

template class GemmPluginProfiler<cublasLtMatmulHeuristicResult_t,
                                  std::shared_ptr<tensorrt_llm::common::CublasMMWrapper>, GemmIdCublas,
                                  GemmIdCublasHash>;

// TODO I dont like the dependency on the MOE plugin here, but MOE needs the
// full context to run profiles template class
// GemmPluginProfiler<tensorrt_llm::cutlass_extensions::CutlassGemmConfig,
// MixtureOfExpertsPlugin*,
//                                   GemmIDMoe, GemmIDMoeHash>;

void CublasLtGemmPluginProfiler::runGemm(const int M, const int N, const int K, const bool transA, const bool transB,
                                         const CublasGemmWrapperPtr &cublasWrapperPtr, const void *act,
                                         const void *weight, void *output,
                                         const std::optional<cublasLtMatmulHeuristicResult_t> &heuristic,
                                         void *workspace, cudaStream_t stream) {
  cublasWrapperPtr->setStream(stream);
  cublasWrapperPtr->setWorkspace(workspace);

  cublasOperation_t transa, transb;
  int m, n, k;
  int lda, ldb, ldc;
  getProblemParams(transa, transb, m, n, k, lda, ldb, ldc, transA, transB, M, N, K);

  cublasWrapperPtr->createDescriptors(transa, transb, m, n, k, lda, ldb, ldc);
  cublasWrapperPtr->Gemm(transa, transb, m, n, k, weight, lda, act, ldb, output, ldc, heuristic);
  cublasWrapperPtr->destroyDescriptors();
}

void CublasLtGemmPluginProfiler::getProblemParams(cublasOperation_t &transa, cublasOperation_t &transb, int &m, int &n,
                                                  int &k, int &lda, int &ldb, int &ldc, int transA, int transB, int M,
                                                  int N, int K) {
  transa = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
  transb = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  m = N;
  n = M;
  k = K;
  lda = transB ? K : N;
  ldb = transA ? M : K;
  ldc = N;
}

int32_t CublasLtGemmPluginProfiler::computeMDimension(int transA, const int32_t nbDims, const int64_t *dims) {
  int32_t M = 1;
  if (transA) {
    for (int i = nbDims - 1; i > 0; --i) {
      M *= dims[i];
    }
  } else {
    for (int i = 0; i < nbDims - 1; ++i) {
      M *= dims[i];
    }
  }
  return M;
}

int32_t CublasLtGemmPluginProfiler::computeNDimension(int transB, const int32_t nbDims, const int64_t *dims) {
  int32_t N = 1;
  if (transB) {
    for (int i = 0; i < nbDims - 1; ++i) {
      N *= dims[i];
    }
  } else {
    for (int i = nbDims - 1; i > 0; --i) {
      N *= dims[i];
    }
  }
  return N;
}

void CublasLtGemmPluginProfiler::runTactic(int m, int n, int k, const CublasLtGemmPluginProfiler::Config &tactic,
                                           char *workspace, const cudaStream_t &stream) {
  size_t dataSize = sizeof(half);
  if (mType == CublasDataType::FLOAT_DATATYPE) {
    dataSize = sizeof(float);
  }

  void *actPtr = reinterpret_cast<void *>(workspace);
  void *weightPtr = reinterpret_cast<void *>(
      nextWorkspacePtrWithAlignment(reinterpret_cast<int8_t *>(actPtr), m * k * dataSize, ALIGNMENT));
  void *outputPtr = reinterpret_cast<void *>(
      nextWorkspacePtrWithAlignment(reinterpret_cast<int8_t *>(weightPtr), n * k * dataSize, ALIGNMENT));
  char *workspacePtr = reinterpret_cast<char *>(
      nextWorkspacePtrWithAlignment(reinterpret_cast<int8_t *>(outputPtr), m * n * dataSize, ALIGNMENT));
  runGemm(m, n, k, mTransA, mTransB, mRunner, actPtr, weightPtr, outputPtr, {tactic}, workspacePtr, stream);
}

bool CublasLtGemmPluginProfiler::checkTactic(int m, int n, int k, const Config &tactic) const {
  cublasOperation_t transa, transb;
  int M = m, N = n, K = k;
  int lda, ldb, ldc;
  getProblemParams(transa, transb, m, n, k, lda, ldb, ldc, mTransA, mTransB, M, N, K);

  mRunner->createDescriptors(transa, transb, m, n, k, lda, ldb, ldc);

  const auto checkResult = mRunner->checkTactic(transa, transb, m, n, k, lda, ldb, ldc, tactic.algo);

  mRunner->destroyDescriptors();

  return checkResult;
}

void CublasLtGemmPluginProfiler::computeTmpSize(int maxM, int n, int k) {
  size_t dataSize = typeSize(mType);
  size_t outputDataSize = typeSize(mOutputType);

  std::vector<size_t> workspaces = {
      maxM * k * dataSize,        // A
      n * k * dataSize,           // B
      maxM * n * outputDataSize,  // C
      CUBLAS_WORKSPACE_SIZE       // workspace
  };
  size_t bytes = calculateTotalWorkspaceSize(workspaces.data(), workspaces.size(), ALIGNMENT);
  setTmpWorkspaceSizeInBytes(bytes);
}

std::vector<CublasLtGemmPluginProfiler::Config> CublasLtGemmPluginProfiler::getTactics(int M, int N, int K) const {
  cublasOperation_t transa, transb;
  int m, n, k;
  int lda, ldb, ldc;
  getProblemParams(transa, transb, m, n, k, lda, ldb, ldc, mTransA, mTransB, M, N, K);

  mRunner->createDescriptors(transa, transb, m, n, k, lda, ldb, ldc);
  const auto heruistics = mRunner->getTactics(transa, transb, m, n, k, lda, ldb, ldc);
  mRunner->destroyDescriptors();

  return heruistics;
}
}  // namespace gemm
}  // namespace tensorrt_llm