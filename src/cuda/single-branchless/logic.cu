#include <stdio.h>
#include "../check.cuh"
#include "kernels.cuh"

static unsigned char* d_grid_in;
static unsigned char* d_grid_out;
static bool initialized = false;

void next(Universe* u_in, Universe* u_out, int generation){

    size_t grid_size = u_in->width * u_in->height * sizeof(unsigned char);

    if(!initialized){
        CHECK(cudaMalloc(&d_grid_in, grid_size));
        CHECK(cudaMalloc(&d_grid_out, grid_size));
        CHECK(cudaMemcpy(d_grid_in, u_in->cells, grid_size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_grid_out, u_in->cells, grid_size, cudaMemcpyHostToDevice));
        initialized = true;
    }
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((u_in->width + block.x - 1) / block.x, 
            (u_in->height + block.y - 1) / block.y);

    kernel<<<grid, block>>>(d_grid_in, d_grid_out, u_in->width, u_in->height, generation);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(d_grid_in, d_grid_out, grid_size, cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(u_out->cells, d_grid_out, grid_size, cudaMemcpyDeviceToHost));
}