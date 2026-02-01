#ifndef CUDA_DEVICE_FUNCS
#define CUDA_DEVICE_FUNCS

#include "cuda_context.cuh"

__device__ int calculate_priority(int x, int y, int generation, int width);

__device__ unsigned char device_grid_get(unsigned char* grid, int x, int y, int width, int height);

__device__ int device_out_of_bounds(int x, int y, int width, int height);

__device__ Proposal create_proposal(int src_x, int src_y, int dest_x, int dest_y, unsigned char type, unsigned char preference, int priority, int is_swap, unsigned char swap_with);

__device__ void generate_sand_proposals(unsigned char* grid, Proposal* proposals, int* count, int x, int y, int generation, int width, int height);

__device__ void generate_water_proposals(unsigned char* grid, Proposal* proposals, int* count, int x, int y, int generation, int width, int height);

#endif