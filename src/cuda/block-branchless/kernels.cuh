#ifndef CUDA_KERNELS
#define CUDA_KERNELS

#include "../../simulation.h"

#define BLOCK_SIZE 16

__global__ void kernel_opt(unsigned char* __restrict__ grid_in, unsigned char* __restrict__ grid_out, int width, int height, int generation);

#endif
