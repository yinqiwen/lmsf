#include <cuda_runtime_api.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include "tops/c_api/c_api.h"

void test_softmax() {
  std::vector<CTensorView> ops;
  CTensorView input;
  input.dtype = ScalarType::DATA_F32;
  input.shape[0] = 3;
  input.shape[1] = 4;
  input.ndim = 2;
  input.stride[0] = 4;
  input.stride[1] = 1;
  std::vector<float> data{0.0226, 0.7511, 0.4131, 0.5771, 0.1227, 0.1021,
                          0.3893, 0.5257, 0.9848, 0.6543, 0.7536, 0.5954};
  float* input_dptr = nullptr;
  cudaMalloc((void**)&input_dptr, sizeof(float) * data.size());
  cudaMemcpy(input_dptr, data.data(), sizeof(float) * data.size(), cudaMemcpyKind::cudaMemcpyHostToDevice);
  input.ptr = input_dptr;

  float* output_dptr = nullptr;
  cudaMalloc((void**)&output_dptr, sizeof(float) * data.size());
  CTensorView output;
  output.dtype = ScalarType::DATA_F32;
  output.ptr = output_dptr;

  cuda_softmax_tensor(input, 0, 0, output);

  float pp[12];
  cudaMemcpy(pp, output_dptr, sizeof(float) * 12, cudaMemcpyKind::cudaMemcpyDeviceToHost);

  printf("test_softmax result:");
  for (int i = 0; i < 12; i++) {
    printf("%f ", pp[i]);
  }
  printf("\n");
}

void test_log_softmax() {
  std::vector<CTensorView> ops;
  CTensorView input;
  input.dtype = ScalarType::DATA_F32;
  input.shape[0] = 3;
  input.shape[1] = 4;
  input.ndim = 2;
  input.stride[0] = 4;
  input.stride[1] = 1;
  std::vector<float> data{0.0226, 0.7511, 0.4131, 0.5771, 0.1227, 0.1021,
                          0.3893, 0.5257, 0.9848, 0.6543, 0.7536, 0.5954};
  float* input_dptr = nullptr;
  cudaMalloc((void**)&input_dptr, sizeof(float) * data.size());
  cudaMemcpy(input_dptr, data.data(), sizeof(float) * data.size(), cudaMemcpyKind::cudaMemcpyHostToDevice);
  input.ptr = input_dptr;

  float* output_dptr = nullptr;
  cudaMalloc((void**)&output_dptr, sizeof(float) * data.size());
  CTensorView output;
  output.dtype = ScalarType::DATA_F32;
  output.ptr = output_dptr;

  cuda_softmax_tensor(input, 1, 0, output);

  float pp[12];
  cudaMemcpy(pp, output_dptr, sizeof(float) * 12, cudaMemcpyKind::cudaMemcpyDeviceToHost);

  printf("test_log_softmax result:");
  for (int i = 0; i < 12; i++) {
    printf("%f ", pp[i]);
  }
  printf("\n");
}

int main() {
  test_softmax();
  test_log_softmax();
  return 0;
}