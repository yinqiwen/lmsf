#include <cuda_runtime_api.h>
#include <stdio.h>
#include <vector>
#include "tops/c_api/c_api.h"

int main() {
  int64_t elem_cnt = 8;
  float* dptr = nullptr;
  cudaMalloc((void**)&dptr, sizeof(float) * elem_cnt);

  CTensorView view;
  view.dtype = ScalarType::DATA_F32;
  view.ptr = dptr;
  view.shape[0] = elem_cnt;
  view.shape[1] = 1;
  view.shape[2] = 1;
  view.shape[3] = 1;

  cuda_reset_random_seed(0);

  cuda_create_exponential_tensor(1.0, nullptr, view);
  float pp[elem_cnt];
  cudaMemcpy(pp, dptr, sizeof(float) * elem_cnt, cudaMemcpyKind::cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  for (int i = 0; i < elem_cnt; i++) {
    printf("%f ", pp[i]);
  }
  printf("\n");

  return 0;
}