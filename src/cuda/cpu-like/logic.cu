#include "kernels.cuh"
#include "../../simulation.h"
void next(Universe* u_in, Universe* u_out, int generation){
    int w = u_in->width;
    int h = u_in->height;
    size_t size = w * h * sizeof(unsigned char);

    unsigned char *d_in, *d_out, *d_clock;

    CHECK(cudaMalloc(&d_in, size));
    CHECK(cudaMalloc(&d_out, size));
    CHECK(cudaMalloc(&d_clock, size));

    CHECK(cudaMemcpy(d_in, u_in->cells, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_out, d_in, size, cudaMemcpyDeviceToDevice));
    CHECK(cudaMemset(d_clock, 0, size));

    dim3 block(256);
    dim3 grid((w + 255) / 256);

    int is_odd = generation % 2 != 0;
    int step_x = 1 - (is_odd * 2);
    int start_x = is_odd * (w - 1);

    for (int y = 0; y < h; y++) {
        row_kernel<<<grid, block>>>(
            d_in, d_out, d_clock,
            w, h,
            y,
            start_x,
            step_x,
            generation
        );
        CHECK(cudaDeviceSynchronize());
    }

    CHECK(cudaMemcpy(u_out->cells, d_out, size, cudaMemcpyDeviceToHost));

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_clock);
}
