#include <stdio.h>
#include "../check.cuh"
#include "kernels.cuh"


void next(Universe* u_in, Universe* u_out, int generation){
    
    int phase = generation % 4;
    int offset_x = (phase == 1 || phase == 3) ? 1 : 0;
    int offset_y = (phase == 1 || phase == 2) ? 1 : 0;

    size_t grid_size = u_in->width * u_in->height * sizeof(unsigned char);

    unsigned char* d_grid_in;
    unsigned char* d_grid_out;
    CHECK(cudaMalloc(&d_grid_in, grid_size));
    CHECK(cudaMalloc(&d_grid_out, grid_size));
    CHECK(cudaMemcpy(d_grid_in, u_in->cells, grid_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_grid_out, u_in->cells, grid_size, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((u_in->width / 2 + block.x - 1) / block.x,
              (u_in->height /2 + block.y - 1) / block.y);

    kernel<<<grid, block>>>(d_grid_in, d_grid_out, u_in->width, u_in->height, offset_x, offset_y, generation);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(u_out->cells, d_grid_out, grid_size, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_grid_in));
    CHECK(cudaFree(d_grid_out));
}