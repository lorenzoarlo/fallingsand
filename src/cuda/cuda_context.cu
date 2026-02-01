#include "cuda_context.cuh"
#include "check.cuh"

CudaContext* cuda_context_create(int width, int height){
    CudaContext* ctx = (CudaContext*) malloc(sizeof(CudaContext));
    if(!ctx){
        fprintf(stderr, "Failed to allocate CudaContext\n");
        return NULL;
    }

    ctx->width = width;
    ctx->height = height; 
    long int total_cells = width * height;

    size_t total_size = total_cells * sizeof(unsigned char);
    CHECK(cudaMalloc(&ctx->d_grid_in, total_size));
    CHECK(cudaMalloc(&ctx->d_grid_out, total_size));

    size_t proposal_size = total_cells * MAX_PROPOSALS_PER_CELL * sizeof(Proposal);
    CHECK(cudaMalloc(&ctx->d_proposals, proposal_size));
    CHECK(cudaMalloc(&ctx->d_prop_counts, total_size));

    size_t cellstate_size = total_cells * sizeof(CellState);
    CHECK(cudaMalloc(&ctx->d_cell_states, cellstate_size));
    
    CHECK(cudaMalloc(&ctx->d_satisfied, total_size));
    CHECK(cudaMalloc(&ctx->d_changed, sizeof(int)));
    CHECK(cudaMalloc(&ctx->d_swap_sources, total_size));

    return ctx;
}

void cuda_init_context(CudaContext* ctx){
    if(!ctx) return;

    long int total_cells = ctx->width * ctx->height;
    size_t total_size = total_cells * sizeof(unsigned char);
    CHECK(cudaMemset(ctx->d_grid_in, P_EMPTY, total_size));
    CHECK(cudaMemset(ctx->d_grid_out, P_EMPTY, total_size));

    size_t proposal_size = total_cells * MAX_PROPOSALS_PER_CELL * sizeof(Proposal);
    CHECK(cudaMemset(ctx->d_proposals, 0, proposal_size));
    CHECK(cudaMemset(ctx->d_prop_counts, 0, total_size));

    size_t cellstate_size = total_cells * sizeof(CellState);
    CHECK(cudaMemset(ctx->d_cell_states, 0, cellstate_size));
    CHECK(cudaMemset(ctx->d_satisfied, 0, total_size));
    CHECK(cudaMemset(ctx->d_swap_sources, 0, total_size));
}

void cuda_context_destroy(CudaContext* ctx){
    if(!ctx) return;

    if(ctx->d_grid_in)      cudaFree(ctx->d_grid_in);
    if(ctx->d_grid_out)     cudaFree(ctx->d_grid_out);
    if(ctx->d_proposals)    cudaFree(ctx->d_proposals);
    if(ctx->d_prop_counts)  cudaFree(ctx->d_prop_counts);
    if(ctx->d_cell_states)  cudaFree(ctx->d_cell_states);
    if(ctx->d_satisfied)    cudaFree(ctx->d_satisfied);
    if(ctx->d_changed)      cudaFree(ctx->d_changed);
    if(ctx->d_swap_sources) cudaFree(ctx->d_swap_sources);

    free(ctx);
}

void cuda_upload_grid(CudaContext* ctx, Universe* u){
    size_t total_size = ctx->width * ctx->height * sizeof(unsigned char);
    CHECK(cudaMemcpy(ctx->d_grid_in, u->cells, total_size, cudaMemcpyHostToDevice));
}

void cuda_download_grid(CudaContext* ctx, Universe* out){
    size_t total_size = ctx->width * ctx->height * sizeof(unsigned char);
    CHECK(cudaMemcpy(out->cells, ctx->d_grid_out, total_size, cudaMemcpyDeviceToHost));
}