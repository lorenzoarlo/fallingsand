#ifndef CUDA_KERNELS
#define CUDA_KERNELS

#include "../../simulation.h"

#define BLOCK_SIZE 32

__global__ void kernel(unsigned char* grid_in, unsigned char* grid_out, int width, int height, int offset_x, int offset_y, int generation);

#endif
