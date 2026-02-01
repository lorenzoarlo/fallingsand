#include "kernels.cuh"

__global__ void generate_proposals(unsigned char* grid_in, Proposal* proposals, unsigned char* prop_counts, int width, int height, int generation){

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cells = width * height;
    if(thread_id >= total_cells) return;
    int x = thread_id % width;
    int y = thread_id / width;
    unsigned char cell = grid_in[thread_id];
    int base_idx = thread_id * MAX_CANDIDATES_PER_CELL;
    int count = 0;

    switch(cell){
        case P_SAND: 
            generate_sand_proposals(grid_in, &proposals[base_idx], &count, x, y, width, height, generation); 
            break;
        case P_WATER: 
            generate_water_proposals(grid_in, &proposals[base_idx], &count, x, y, width, height, generation); 
            break;
        case P_WALL: 
            proposals[base_idx] = create_proposal(x, y, x, y, P_WALL, 0, calculate_priority(x, y, generation, width), 0, P_EMPTY); 
            count = 0; 
            break;
        case P_EMPTY:
            count = 0;
            break;
        default:
            count = 0;
            break;
    }

    prop_counts[thread_id] = count;

}
__global__ void collect_proposals(Proposal* proposals, unsigned char* prop_counts, CellState* cell_states, int width, int height){

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cells = width * height;
    if(thread_id >= total_cells) return;
    int dest_x = thread_id % width;
    int dest_y = thread_id / width;
    CellState* cell_state = &cell_states[thread_id];
    cell_state->candidate_count = 0;
    cell_state->resolved = 0;
    cell_state->winner_is_swap = 0;

    for(int sy = dest_y - 1; sy <= dest_y + 1; sy++){
        for(int sx = dest_x - 1; sx <= dest_x + 1; sx++){
            if(sx < 0 || sx >= width || sy < 0 || sy >= height) continue;
            int source_idx = sy * width + sx;
            int proposal_count = prop_counts[source_idx];
            int base_idx = source_idx * MAX_CANDIDATES_PER_CELL;
            for(int p = 0; p < proposal_count; p++){
                Proposal* prop = &proposals[base_idx + p];
                if(prop->dest_x == dest_x && prop->dest_y == dest_y){
                    int cand_idx = cell_state->candidate_count;
                    if(cand_idx < MAX_CANDIDATES_PER_CELL){
                        cell_state->candidates[cand_idx] = *prop;
                        cell_state->candidate_count++;
                    }
                }
            }
        }
    }

}
__global__ void resolv_conflicts(CellState* cell_states, unsigned char* satisfied, unsigned char current_pref, int width, int height, int* changed){
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cells = width * height;
    if(thread_id >= total_cells) return;
    CellState* cell_state = &cell_states[thread_id];
    if(cell_state->resolved) return;

    Proposal* best_proposal = nullptr;
    for(int i = 0; i < cell_state->candidate_count; i++){
        Proposal* prop = &cell_state->candidates[i];
        if(prop->preference == current_pref && !satisfied[prop->src_y * width + prop->src_x]){
            if(best_proposal == nullptr || prop->priority < best_proposal->priority){
                best_proposal = prop;
            }
        }
    }

    if(best_proposal != nullptr){
        cell_state->final_type = best_proposal->type;
        cell_state->winner_src_x = best_proposal->src_x;
        cell_state->winner_src_y = best_proposal->src_y;
        cell_state->winner_is_swap = best_proposal->is_swap;
        cell_state->resolved = 1;
        satisfied[best_proposal->src_y * width + best_proposal->src_x] = 1;
        atomicExch(changed, 1);
    }
}
__global__ void mark_swaps(CellState* cell_states, unsigned char* swap_sources, int width, int height){
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cells = width * height;
    if(thread_id >= total_cells) return;
    CellState* cell_state = &cell_states[thread_id];
    if(cell_state->resolved && cell_state->winner_is_swap){
        int src_idx = cell_state->winner_src_y * width + cell_state->winner_src_x;
        swap_sources[src_idx] = P_WATER;
    }
}
__global__ void apply_state(CellState* cell_states, unsigned char* swap_sources, unsigned char* grid_out, int width, int height){
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cells = width * height;
    if(thread_id >= total_cells) return;
    CellState* cell_state = &cell_states[thread_id];
    if(swap_sources[thread_id] == P_WATER){
        grid_out[thread_id] = P_WATER;
        return;
    }
    if(cell_state->resolved){
        grid_out[thread_id] = cell_state->final_type;
    } else {
        grid_out[thread_id] = P_EMPTY;
    }
}