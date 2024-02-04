#include <cuda_runtime_api.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include "tops/arrayfire/common/mem.h"
#include "tops/c_api/c_api.h"
#include "tops/torch/common/tensor_iterator.h"

void test_iter() {
  std::vector<CTensorView> ops;
  CTensorView src;
  src.dtype = ScalarType::DATA_I64;
  src.shape[0] = 1;
  src.shape[1] = 8;
  src.ndim = 2;
  src.stride[0] = 8;
  src.stride[1] = 0;
  ops.push_back(src);
  src.stride[0] = 8;
  src.stride[1] = 1;
  ops.push_back(src);
  ops.push_back(src);
  at::native::TensorIterator iter(ops);

  for (auto v : iter.shape()) {
    printf("%lld ", v);
  }
  printf("\n");

  printf("%lld %lld\n", *iter.strides(0), *(iter.strides(0) + 1));
  printf("%lld %lld\n", *iter.strides(1), *(iter.strides(1) + 1));
  printf("%lld %lld\n", *iter.strides(2), *(iter.strides(2) + 1));
  printf("\n");
}

void test_scatter0() {
  std::vector<int64_t> src_data = {0, 1, 2, 3, 4, 5, 6, 7};
  int64_t* src_dptr = nullptr;
  cudaMalloc((void**)&src_dptr, sizeof(int64_t) * src_data.size());
  cudaMemcpy(src_dptr, src_data.data(), sizeof(int64_t) * src_data.size(), cudaMemcpyKind::cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  CTensorView src;
  src.dtype = ScalarType::DATA_I64;
  src.ptr = src_dptr;
  src.shape[0] = 1;
  src.shape[1] = 8;
  src.shape[2] = 1;
  src.shape[3] = 1;
  src.stride[0] = 8;
  src.stride[1] = 1;
  src.stride[2] = 1;
  src.stride[3] = 1;
  src.ndim = 2;

  std::vector<int64_t> index_data = {0, 1, 2, 3, 4, 5, 6, 7};
  int64_t* index_dptr = nullptr;
  cudaMalloc((void**)&index_dptr, sizeof(int64_t) * index_data.size());
  cudaMemcpy(index_dptr, index_data.data(), sizeof(int64_t) * index_data.size(),
             cudaMemcpyKind::cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  CTensorView index;
  index.dtype = ScalarType::DATA_I64;
  index.ptr = index_dptr;
  // printf("###index_dptr:%p\n", index_dptr);
  index.shape[0] = 1;
  index.shape[1] = 8;
  index.shape[2] = 1;
  index.shape[3] = 1;
  index.stride[0] = 8;
  index.stride[1] = 1;
  index.stride[2] = 1;
  index.stride[3] = 1;
  index.ndim = 2;

  int64_t* dst_dptr = nullptr;
  cudaMalloc((void**)&dst_dptr, sizeof(int64_t) * 8);
  cudaMemset(dst_dptr, 0, sizeof(int64_t) * 8);
  CTensorView dst;
  dst.dtype = ScalarType::DATA_I64;
  dst.ptr = dst_dptr;
  dst.shape[0] = 1;
  dst.shape[1] = 8;
  dst.shape[2] = 1;
  dst.shape[3] = 1;
  dst.stride[0] = 8;
  dst.stride[1] = 1;
  dst.stride[2] = 1;
  dst.stride[3] = 1;
  dst.ndim = 2;

  cuda_scatter_tensor(index, src, 1, dst, 0);

  cudaDeviceSynchronize();
  int64_t pp[8];
  cudaMemcpy(pp, dst_dptr, sizeof(int64_t) * 8, cudaMemcpyKind::cudaMemcpyDeviceToHost);

  printf("scatter0 result:");
  for (int i = 0; i < 8; i++) {
    printf("%lld ", pp[i]);
  }
  printf("\n");
}
void test_scatter1() {
  // >>> src = torch.arange(1, 11).reshape((2, 5))
  // >>> src
  // tensor([[ 1,  2,  3,  4,  5],
  //         [ 6,  7,  8,  9, 10]])
  // >>> index = torch.tensor([[0, 1, 2, 0]])
  // >>> torch.zeros(3, 5, dtype=src.dtype).scatter_(0, index, src)
  // tensor([[1, 0, 0, 4, 0],
  //         [0, 2, 0, 0, 0],
  //         [0, 0, 3, 0, 0]])

  std::vector<uint32_t> src_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int64_t* src_dptr = nullptr;
  cudaMalloc((void**)&src_dptr, sizeof(uint32_t) * src_data.size());
  cudaMemcpy(src_dptr, src_data.data(), sizeof(uint32_t) * src_data.size(), cudaMemcpyKind::cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  CTensorView src;
  src.dtype = ScalarType::DATA_U32;
  src.ptr = src_dptr;
  src.shape[0] = 2;
  src.shape[1] = 5;
  src.shape[2] = 1;
  src.shape[3] = 1;
  src.stride[0] = 5;
  src.stride[1] = 1;
  src.stride[2] = 1;
  src.stride[3] = 1;
  src.ndim = 2;

  std::vector<uint32_t> index_data = {0, 1, 2, 0};
  int64_t* index_dptr = nullptr;
  cudaMalloc((void**)&index_dptr, sizeof(uint32_t) * index_data.size());
  cudaMemcpy(index_dptr, index_data.data(), sizeof(uint32_t) * index_data.size(),
             cudaMemcpyKind::cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  CTensorView index;
  index.dtype = ScalarType::DATA_U32;
  index.ptr = index_dptr;
  // printf("###index_dptr:%p\n", index_dptr);
  index.shape[0] = 1;
  index.shape[1] = 4;
  index.shape[2] = 1;
  index.shape[3] = 1;
  index.stride[0] = 4;
  index.stride[1] = 1;
  index.stride[2] = 1;
  index.stride[3] = 1;
  index.ndim = 2;

  uint32_t* dst_dptr = nullptr;
  cudaMalloc((void**)&dst_dptr, sizeof(uint32_t) * 15);
  cudaMemset(dst_dptr, 0, sizeof(uint32_t) * 15);
  CTensorView dst;
  dst.dtype = ScalarType::DATA_U32;
  dst.ptr = dst_dptr;
  dst.shape[0] = 3;
  dst.shape[1] = 5;
  dst.shape[2] = 1;
  dst.shape[3] = 1;
  dst.stride[0] = 5;
  dst.stride[1] = 1;
  dst.stride[2] = 1;
  dst.stride[3] = 1;
  dst.ndim = 2;

  cuda_scatter_tensor(index, src, 0, dst, 0);

  cudaDeviceSynchronize();
  uint32_t pp[15];
  cudaMemcpy(pp, dst_dptr, sizeof(uint32_t) * 15, cudaMemcpyKind::cudaMemcpyDeviceToHost);

  printf("scatter1 result:");
  for (int i = 0; i < 15; i++) {
    printf("%d ", pp[i]);
  }
  printf("\n");
}

int main() {
  // >>> src = torch.arange(1, 11).reshape((2, 5))
  // >>> src
  // tensor([[ 1,  2,  3,  4,  5],
  //         [ 6,  7,  8,  9, 10]])
  // >>> index = torch.tensor([[0, 1, 2, 0]])
  // >>> torch.zeros(3, 5, dtype=src.dtype).scatter_(0, index, src)
  // tensor([[1, 0, 0, 4, 0],
  //         [0, 2, 0, 0, 0],
  //         [0, 0, 3, 0, 0]])

  // std::vector<uint32_t> src_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  // int64_t* src_dptr = nullptr;
  // cudaMalloc((void**)&src_dptr, sizeof(uint32_t) * src_data.size());
  // cudaMemcpy(src_dptr, src_data.data(), sizeof(uint32_t) * src_data.size(), cudaMemcpyKind::cudaMemcpyHostToDevice);
  // cudaDeviceSynchronize();
  // CTensorView src;
  // src.dtype = ScalarType::DATA_U32;
  // src.ptr = src_dptr;
  // src.shape[0] = 2;
  // src.shape[1] = 5;
  // src.shape[2] = 1;
  // src.shape[3] = 1;
  // src.stride[0] = 5;
  // src.stride[1] = 1;
  // src.stride[2] = 1;
  // src.stride[3] = 1;
  // src.ndim = 2;

  // std::vector<uint32_t> index_data = {0, 1, 2, 0};
  // int64_t* index_dptr = nullptr;
  // cudaMalloc((void**)&index_dptr, sizeof(uint32_t) * index_data.size());
  // cudaMemcpy(index_dptr, index_data.data(), sizeof(uint32_t) * index_data.size(),
  //            cudaMemcpyKind::cudaMemcpyHostToDevice);
  // cudaDeviceSynchronize();
  // CTensorView index;
  // index.dtype = ScalarType::DATA_U32;
  // index.ptr = index_dptr;
  // // printf("###index_dptr:%p\n", index_dptr);
  // index.shape[0] = 1;
  // index.shape[1] = 4;
  // index.shape[2] = 1;
  // index.shape[3] = 1;
  // index.stride[0] = 4;
  // index.stride[1] = 1;
  // index.stride[2] = 1;
  // index.stride[3] = 1;
  // index.ndim = 2;

  // uint32_t* dst_dptr = nullptr;
  // cudaMalloc((void**)&dst_dptr, sizeof(uint32_t) * 15);
  // cudaMemset(dst_dptr, 0, sizeof(uint32_t) * 15);
  // CTensorView dst;
  // dst.dtype = ScalarType::DATA_U32;
  // dst.ptr = dst_dptr;
  // dst.shape[0] = 3;
  // dst.shape[1] = 5;
  // dst.shape[2] = 1;
  // dst.shape[3] = 1;
  // dst.stride[0] = 5;
  // dst.stride[1] = 1;
  // dst.stride[2] = 1;
  // dst.stride[3] = 1;
  // dst.ndim = 2;

  // cuda_scatter_tensor(index, src, 0, dst, 0);

  // cudaDeviceSynchronize();
  // uint32_t pp[15];
  // cudaMemcpy(pp, dst_dptr, sizeof(uint32_t) * 15, cudaMemcpyKind::cudaMemcpyDeviceToHost);

  // for (int i = 0; i < 15; i++) {
  //   printf("%d ", pp[i]);
  // }
  // printf("\n");

  // int64_t pp[4];
  // cudaMemcpy(pp, index_dptr, sizeof(int64_t) * 4, cudaMemcpyKind::cudaMemcpyDeviceToHost);
  // cudaDeviceSynchronize();

  // for (int i = 0; i < 4; i++) {
  //   printf("%lld ", pp[i]);
  // }
  // printf("\n");
  // test_iter();

  // test_iter();
  test_scatter0();
  test_scatter1();
  return 0;
}