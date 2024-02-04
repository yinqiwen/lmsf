#include <vector>
#include "tops/arrayfire/common/mem.h"
#include "tops/arrayfire/ops/scan.h"
#include "tops/arrayfire/ops/sort_index.h"
#include "tops/arrayfire/ops/topk.h"

int main() {
  std::vector<float> data = {0.6719, -0.2162, 2.3332, 0.0608,  0.9497, -0.5793,
                             0.6058, 0.0061,  0.3343, -0.5071, 1.0960, 0.9553};
  arrayfire::dim4 dims(4, 3);

  auto array = arrayfire::createArray(data, dims, 0);

  arrayfire::cuda::Array<float> out;
  arrayfire::cuda::Array<uint32_t> indices;
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
  arrayfire::cuda::sort_index(out, indices, array, 0, true, &props, 0);
  printf("sorted:%s\n", out.to_string().c_str());
  printf("indices:%s\n", indices.to_string().c_str());

  out.reset();
  indices.reset();
  arrayfire::cuda::sort_index(out, indices, array, 1, true, &props, 0);
  printf("sorted:%s\n", out.to_string().c_str());
  printf("indices:%s\n", indices.to_string().c_str());

  arrayfire::dim4 topk_dims(data.size() / 2, 2);
  auto topk_in = arrayfire::createArray(data, topk_dims, 0);
  arrayfire::cuda::Array<float> topk_value;
  arrayfire::cuda::Array<uint32_t> topk_indices;
  arrayfire::cuda::topk(topk_value, topk_indices, topk_in, 3, 0, arrayfire::AF_TOPK_MAX, 0);

  printf("topk:%s\n", topk_value.to_string().c_str());
  printf("topk indices:%s\n", topk_indices.to_string().c_str());

  std::vector<float> cumsum_data = {-0.8286, -0.4890, 0.5155, 0.8443,  0.1865,
                                    -0.1752, -2.0595, 0.1850, -1.1571, -0.4243};

  arrayfire::cuda::Array<float> cumsum_out;
  arrayfire::dim4 cumsum_dims(5, 2);
  auto cumsum_array = arrayfire::createArray(cumsum_data, cumsum_dims, 0);
  arrayfire::cuda::scan<af_add_t, float, float>(cumsum_out, cumsum_array, 0, true, &props, 0);
  printf("cumsum:%s\n", cumsum_out.to_string().c_str());
  return 0;
}