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

#pragma once
#include <cuda_runtime_api.h>
#include <stdint.h>
extern "C" {
typedef enum {
  DATA_U8 = 0,
  DATA_F16,
  DATA_BF16,
  DATA_F32,
  DATA_F64,
  DATA_U32,
  DATA_I64,
} ScalarType;

struct CShapeView {
  int64_t shape[4];
  int ndim;
};

struct CTensorView {
  void *ptr = nullptr;
  int64_t shape[4];
  int64_t stride[4];
  uint32_t dtype;
  int ndim;
};

cudaDeviceProp *getCudaDeviceProp();
void *get_temp_buffer(size_t n);

size_t element_size(ScalarType type);
bool is_tensor_contiguous(const CTensorView *t);

void cuda_reset_random_seed(uint64_t seed);

size_t get_tensor_element_count(const CTensorView *tensor);

void cuda_sort_tensor(CTensorView input, uint32_t dim, bool ascend, cudaStream_t stream, CTensorView output,
                      CTensorView indices);
void cuda_argsort_tensor(CTensorView input, bool ascend, CTensorView indices, cudaStream_t stream);
void cuda_dim_gather_tensor(CTensorView input, int dim, CTensorView index, cudaStream_t stream, CTensorView output);

void cuda_cumsum_tensor(CTensorView input, uint32_t dim, cudaStream_t stream, CTensorView output);

void cuda_topk_tensor(CTensorView input, int k, int dim, int topk_type, cudaStream_t stream, CTensorView output,
                      CTensorView indices);

void cuda_create_exponential_tensor(float lambd, cudaStream_t stream, CTensorView output);

void cuda_repeat_tensor(CTensorView input, uint32_t dim0, uint32_t dim1, uint32_t dim2, uint32_t dim3,
                        cudaStream_t stream, CTensorView output);

void cuda_async_htod(void *dptr, const void *hptr, int64_t n, cudaStream_t stream);
void cuda_async_dtod(void *dptr, const void *hptr, int64_t n, cudaStream_t stream);

void cuda_async_set(void *dptr, int v, int n, cudaStream_t stream);

void cuda_scatter_tensor(CTensorView index, CTensorView src, int64_t dim, CTensorView dst, cudaStream_t stream);

void cuda_softmax_tensor(CTensorView input, int algorithm, cudaStream_t stream, CTensorView output);

void cuda_oneflow_rms_norm(CTensorView x, CTensorView weight, CShapeView normalized_shape, float epsilon,
                           cudaStream_t stream, CTensorView inv_rms, CTensorView y);

void cuda_arrange_int_tensor(int64_t start, int64_t delta, const int64_t arange_elem_cnt, CTensorView out,
                             cudaStream_t stream);
void cuda_arrange_float_tensor(double start, double delta, const int64_t arange_elem_cnt, CTensorView out,
                               cudaStream_t stream);

void fastertransformer_silu_activation(CTensorView a, CTensorView b, int num_token, int inter_size,
                                       cudaStream_t stream);

// int load_gemm_config(const char *file);
void gemm_config(void *gemm, int dtype, bool transA, bool transB, CShapeView min_input, CShapeView max_input,
                 CShapeView weight);
int save_gemm_config();
void *new_gemm(int device, cudaStream_t stream, int dtype);
void delete_gemm(void *gemm);

int gemm_execute(void *gemm, int transa, int transb, CTensorView input, CTensorView weight, CTensorView output);
}