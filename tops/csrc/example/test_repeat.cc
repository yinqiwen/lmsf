#include <cuda_runtime_api.h>
#include <stdio.h>
#include <vector>
#include "tops/arrayfire/common/mem.h"
#include "tops/c_api/c_api.h"

int main() {
  std::vector<float> data = {1.0, 2.0, 3.0};
  arrayfire::dim4 dims(3, 1);
  arrayfire::dim4 repeat_dims(2, 4);

  auto array = arrayfire::createArray(data, dims, 0);
  arrayfire::dim4 out_dims = dims * repeat_dims;
  auto output = arrayfire::createEmptyArray<float>(out_dims);

  CTensorView in;
  in.dtype = ScalarType::DATA_F32;
  in.ptr = array.get();
  in.shape[0] = dims.dims[0];
  in.shape[1] = 1;
  in.shape[2] = 1;
  in.shape[3] = 1;

  CTensorView out;
  out.dtype = ScalarType::DATA_F32;
  out.ptr = output.get();
  out.shape[0] = out_dims.dims[0];
  out.shape[1] = out_dims.dims[1];
  out.shape[2] = 1;
  out.shape[3] = 1;
  cuda_repeat_tensor(in, repeat_dims.dims[0], repeat_dims.dims[1], repeat_dims.dims[2], repeat_dims.dims[3], 0, out);

  printf("%s\n", output.to_string().c_str());
  return 0;
}