#include "kernels.cuh"
#include "../../simulation.h"

void next(Universe* u_in, Universe* u_out, int generation){
    size_t total_size = u_in->width * u_in->height * sizeof(unsigned char);
    unsigned char* d_in;
    unsigned char* d_out;
    unsigned char* d_clock; 
    CHECK(cudaMalloc((void**)&d_in, total_size));
    CHECK(cudaMalloc((void**)&d_out, total_size));
    CHECK(cudaMalloc((void**)&d_clock, total_size));
    CHECK(cudaMemcpy(d_in, u_in->cells, total_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_out, d_in, total_size, cudaMemcpyDeviceToDevice));
    CHECK(cudaMemset(d_clock, 0, total_size));
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((u_in->width + block.x - 1) / block.x,
            (u_in->height + block.y - 1) / block.y);
    sand_step_kernel<<<grid, block>>>(d_in, d_out, u_in->width, u_in->height, generation, d_clock);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(u_out->cells, d_out, total_size, cudaMemcpyDeviceToHost));
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_clock);
}