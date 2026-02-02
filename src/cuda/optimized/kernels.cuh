#ifndef CUDA_KERNELS
#define CUDA_KERNELS

#include "../check.cuh"
#include <stdio.h>
#include "../../universe.h"
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void sand_step_kernel(const unsigned char* in, unsigned char* out, int w, int h, int gen, unsigned char* clock);

#endif