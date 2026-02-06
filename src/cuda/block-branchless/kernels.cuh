#ifndef CUDA_KERNELS
#define CUDA_KERNELS

#include "../../simulation.h"

#define BLOCK_SIZE 32

__global__ void kernel_opt(unsigned char* grid_in, unsigned char* grid_out, int width, int height, int generation);

#endif
