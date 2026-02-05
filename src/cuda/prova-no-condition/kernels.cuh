#ifndef CUDA_KERNELS
#define CUDA_KERNELS

#include "../../simulation.h"

#define BLOCK_SIZE 32

#define SAND_NOISE_CHANCE 0.4f
#define WATER_FALL_DOWN_CHANCE 0.9f
#define WATER_FALL_DENSITY_CHANCE 1.0f
#define WATER_MOVE_DIAGONAL_CHANCE 0.5f
#define WATER_MOVE_HORIZONTAL_CHANCE 0.8f

#define COND_SWAP(cond, a, b) do { \
    unsigned char mask = (unsigned char)(-(int)(cond)); \
    unsigned char xor_val = (*(a) ^ *(b)) & mask; \
    *(a) ^= xor_val; \
    *(b) ^= xor_val; \
} while(0)

__global__ void kernel(unsigned char* grid_in, unsigned char* grid_out, int width, int height, int generation);

#endif
