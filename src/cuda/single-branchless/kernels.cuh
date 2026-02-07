#ifndef CUDA_KERNELS
#define CUDA_KERNELS

#include "../../simulation.h"

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

#define TILE_DIM_X (BLOCK_SIZE_X + 2)
#define TILE_DIM_Y (BLOCK_SIZE_Y + 2)

__global__ void kernel(unsigned char* grid_in, unsigned char* grid_out, int width, int height, int generation);

#endif
