#ifndef CUDA_KERNELS
#define CUDA_KERNELS

#include "../../simulation.h"

#define BLOCK_SIZE 32

#define SAND_NOISE_CHANCE 0.4f
#define WATER_FALL_DOWN_CHANCE 0.9f
#define WATER_FALL_DENSITY_CHANCE 1.0f
#define WATER_MOVE_DIAGONAL_CHANCE 0.5f
#define WATER_MOVE_HORIZONTAL_CHANCE 0.8f

__global__ void kernel(unsigned char* grid_in, unsigned char* grid_out, int width, int height, int offset_x, int offset_y, int generation);

#endif