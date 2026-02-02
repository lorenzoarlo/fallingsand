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

    // 1. Generate proposals
    generate_proposals<<<grid, block>>>(ctx->d_grid_in, ctx->d_proposals, ctx->d_prop_counts, ctx->width, ctx->height, generation);
    CHECK(cudaDeviceSynchronize());

    // 2. Collect proposals
    collect_proposals<<<grid, block>>>(ctx->d_proposals, ctx->d_prop_counts, ctx->d_cell_states, ctx->width, ctx->height);
    CHECK(cudaDeviceSynchronize());

    // 3. Initialize satisfied buffer
    CHECK(cudaMemset(ctx->d_satisfied, 0, total_cells * sizeof(unsigned char)));
    
    // 4. Resolve conflicts
    for(int i = 0; i < MAX_PROPOSALS_PER_CELL; i++){
        int h_changed = 0;
        CHECK(cudaMemset(ctx->d_changed, 0, sizeof(int)));
        resolv_conflicts<<<grid, block>>>(ctx->d_cell_states, ctx->d_satisfied, (unsigned char)i, ctx->width, ctx->height, ctx->d_changed);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(&h_changed, ctx->d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        if(h_changed == 0) break;
    }
    CHECK(cudaMemset(ctx->d_swap_buffer, 0, total_cells * sizeof(unsigned char)));
    // 5. Marca le destinazioni degli swap
    mark_swap_destinations<<<grid, block>>>(ctx->d_cell_states, ctx->d_swap_buffer, ctx->width, ctx->height);
    CHECK(cudaDeviceSynchronize());

    // 6. Applica movimenti normali e swap "in discesa"
    apply_movements<<<grid, block>>>(ctx->d_cell_states, ctx->d_grid_out, ctx->d_grid_in, ctx->d_swap_buffer, ctx->width, ctx->height);
    CHECK(cudaDeviceSynchronize());

    // 7. Completa gli swap facendo "salire" l'acqua
    complete_swaps<<<grid, block>>>(ctx->d_cell_states, ctx->d_grid_out, ctx->d_grid_in, ctx->d_swap_buffer, ctx->width, ctx->height);
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