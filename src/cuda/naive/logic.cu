#include <stdio.h>
#include "cuda_context.cuh"
#include "../check.cuh"
#include "kernels.cuh"
#include "../../simulation.h"

void next_cuda_naive(Universe* u, Universe* out, int generation, CudaContext* ctx){
    int total_cells = u->width * u->height;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((u->width + block.x - 1) / block.x,
              (u->height + block.y - 1) / block.y);

    // First kernel: generate proposals
    generate_proposals<<<grid, block>>>(ctx->d_grid_in, ctx->d_proposals, ctx->d_prop_counts, ctx->width, ctx->height, generation);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // Second kernel: collect proposals
    collect_proposals<<<grid, block>>>(ctx->d_proposals, ctx->d_prop_counts, ctx->d_cell_states, ctx->width, ctx->height);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // Resolve conficts
    for(int i = 0; i < MAX_PROPOSALS_PER_CELL; i++){
        int h_changed = 0;
        CHECK(cudaMemset(ctx->d_changed, 0, sizeof(int)));
        resolv_conflicts<<<grid, block>>>(ctx->d_cell_states, ctx->d_satisfied, (unsigned char)i, ctx->width, ctx->height, ctx->d_changed);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(&h_changed, ctx->d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        if(h_changed == 0) break;
    }

    // Third kernel: mark swaps
    mark_swaps<<<grid, block>>>(ctx->d_cell_states, ctx->d_swap_sources, ctx->width, ctx->height);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // Fourth kernel: apply state changes
    apply_state<<<grid, block>>>(ctx->d_cell_states, ctx->d_swap_sources, ctx->d_grid_out, ctx->width, ctx->height);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

}

void next(Universe* u_in, Universe* u_out, int generation){
    CudaContext* ctx = cuda_context_create(u_in->width, u_in->height);
    cuda_init_context(ctx);
    cuda_upload_grid(ctx, u_in);
    // Compute frame
    next_cuda_naive(u_in, u_out, generation, ctx);
    // Download result
    cuda_download_grid(ctx, u_out);
    cuda_context_destroy(ctx);
}