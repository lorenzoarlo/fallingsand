#ifndef CUDA_KERNELS
#define CUDA_KERNELS

#include "../check.cuh"
#include <stdio.h>
#include "../../universe.h"
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void row_kernel(
    const unsigned char* in,
    unsigned char* out,
    unsigned char* clock,
    int w, int h,
    int y,
    int start_x,
    int step_x,
    int gen);
#endif