#ifndef TT_KERNEL_INCLUDES
#define TT_KERNEL_INCLUDES

#include <cuda.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#endif

void unload_add_kernel_0d1d2d3(void);
void load_add_kernel_0d1d2d3(void);
// tt-linker: add_kernel_0d1d2d3:CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements
CUresultadd_kernel_0d1d2d3(CUstream stream, unsigned int gX, unsigned int gY,
                      unsigned int gZ, unsigned int numWarps, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements);
