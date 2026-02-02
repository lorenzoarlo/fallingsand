#ifndef CUDA_KERNELS
#define CUDA_KERNELS

#include "cuda_context.cuh"

__global__ void generate_proposals(unsigned char* grid_in, Proposal* proposals, unsigned char* prop_counts, int width, int height, int generation);
__global__ void collect_proposals(Proposal* proposals, unsigned char* prop_counts, CellState* cell_states, int width, int height);
__global__ void resolv_conflicts(CellState* cell_states, unsigned char* satisfied, unsigned char current_pref, int width, int height, int* changed);
__global__ void mark_swap_destinations(CellState* cell_states, unsigned char* swap_buffer, int width, int height);
__global__ void apply_movements(CellState* cell_states, unsigned char* grid_out, unsigned char* grid_in, unsigned char* swap_buffer, int width, int height);
__global__ void complete_swaps(CellState* cell_states, unsigned char* grid_out, unsigned char* grid_in, unsigned char* swap_buffer, int width, int height);

#endif