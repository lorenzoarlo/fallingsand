#include "kernels.cuh"

__device__ int device_out_of_bounds(int x, int y, int width, int height){
    return (x < 0 || x >= width || y < 0 || y >= height);
}

__device__ int calculate_priority(int x, int y, int generation, int width){
    if(generation % 2 == 0){
        return y * width + x;
    } else {
        return (width - 1 - x) + (width * y);
    }
}

__device__ unsigned char device_grid_get(unsigned char* grid, int x, int y, int width, int height){
    if(device_out_of_bounds(x, y, width, height)){
        return P_WALL;
    }
    return grid[y * width + x];
}

__device__ Proposal create_proposal(int src_x, int src_y, int dest_x, int dest_y, unsigned char type, unsigned char preference, int priority, unsigned char is_swap, unsigned char swap_with){
    Proposal p;
    p.src_x = src_x;
    p.src_y = src_y;
    p.dest_x = dest_x;
    p.dest_y = dest_y;
    p.type = type;
    p.preference = preference;
    p.priority = priority;
    p.is_swap = is_swap;
    return p;
}

__device__ void generate_sand_proposals(unsigned char* grid, Proposal* proposals, int* count, int x, int y, int generation, int width, int height){
    int priority = calculate_priority(x, y, generation, width);
    unsigned char below = device_grid_get(grid, x, y + 1, width, height);
    int dir = (generation % 2 == 0) ? 1 : -1;
    int diag1_x = x - dir;
    int diag2_x = x + dir;

    int pref = 0;
    
    // Pref 0: fall to empty space below
    if(below == P_EMPTY){
        proposals[*count] = create_proposal(x, y, x, y + 1, P_SAND, pref, priority, 0, P_EMPTY);
        (*count)++;
        pref++;
    }
    else {
        // Pref 0: First diagonal if below is blocked
        unsigned char diag1_below = device_grid_get(grid, diag1_x, y + 1, width, height);
        if(diag1_below == P_EMPTY){
            proposals[*count] = create_proposal(x, y, diag1_x, y + 1, P_SAND, pref, priority, 0, P_EMPTY);
            (*count)++;
        }
        pref++;

        // Pref 1: Second diagonal
        unsigned char diag2_below = device_grid_get(grid, diag2_x, y + 1, width, height);
        if(diag2_below == P_EMPTY){
            proposals[*count] = create_proposal(x, y, diag2_x, y + 1, P_SAND, pref, priority, 0, P_EMPTY);
            (*count)++;
        }
        pref++;

        // Pref 2: Swap with water below (SOLO se non può andare in diagonale)
        if(below == P_WATER){
            proposals[*count] = create_proposal(x, y, x, y + 1, P_SAND, pref, priority, 1, P_EMPTY);
            (*count)++;
            pref++;
        }
    }

    // Ultima preferenza: Stay in place
    proposals[*count] = create_proposal(x, y, x, y, P_SAND, pref, priority, 0, P_EMPTY);
    (*count)++;
}

__device__ void generate_water_proposals(unsigned char* grid, Proposal* proposals, int* count, int x, int y, int generation, int width, int height){
    int priority = calculate_priority(x, y, generation, width);
    unsigned char below = device_grid_get(grid, x, y + 1, width, height);
    int left_x = x - 1;
    int right_x = x + 1;
    int diag_first_x = (generation % 2 == 0) ? left_x : right_x;
    int diag_second_x = (generation % 2 == 0) ? right_x : left_x;
    int horiz_first_x = (generation % 2 == 0) ? right_x : left_x;
    int horiz_second_x = (generation % 2 == 0) ? left_x : right_x;

    int pref = 0;
    
    // Pref 0: fall to empty space below
    if(below == P_EMPTY){
        proposals[*count] = create_proposal(x, y, x, y + 1, P_WATER, (unsigned char) pref, priority, 0, P_EMPTY);
        (*count)++;
    }
    pref++;

    // Pref 1-2: First and second diagonals if below is blocked
    if(below != P_EMPTY){
        // Pref 1: First diagonal
        unsigned char diag_first_below = device_grid_get(grid, diag_first_x, y + 1, width, height);
        if(diag_first_below == P_EMPTY){
            proposals[*count] = create_proposal(x, y, diag_first_x, y + 1, P_WATER, (unsigned char) pref, priority, 0, P_EMPTY);
            (*count)++;
        }
        pref++;
        // Pref 2: Second diagonal
        unsigned char diag_second_below = device_grid_get(grid, diag_second_x, y + 1, width, height);
        if(diag_second_below == P_EMPTY){
            proposals[*count] = create_proposal(x, y, diag_second_x, y + 1, P_WATER, (unsigned char) pref, priority, 0, P_EMPTY);
            (*count)++;
        }
        pref++;
    }

    pref = (pref < 3) ? 3 : pref;

    // Pref 3: First horizontal
    unsigned char horiz_first = device_grid_get(grid, horiz_first_x, y, width, height);
    if(horiz_first == P_EMPTY){
        proposals[*count] = create_proposal(x, y, horiz_first_x, y, P_WATER, (unsigned char) pref, priority, 0, P_EMPTY);
        (*count)++;
    }
    pref++;
    // Pref 4: Second horizontal
    unsigned char horiz_second = device_grid_get(grid, horiz_second_x, y, width, height);
    if(horiz_second == P_EMPTY){
        proposals[*count] = create_proposal(x, y, horiz_second_x, y, P_WATER, (unsigned char) pref, priority, 0, P_EMPTY);
        (*count)++;
    }
    pref++;

    // Pref 5: Stay in place
    proposals[*count] = create_proposal(x, y, x, y, P_WATER, (unsigned char) pref, priority, 0, P_EMPTY);
    (*count)++;

}

__global__ void generate_proposals(unsigned char* grid_in, Proposal* proposals, unsigned char* prop_counts, int width, int height, int generation){
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    if(tid_x >= width || tid_y >= height) return;
    int thread_id = tid_y * width + tid_x;
    int total_cells = width * height;
    if(thread_id >= total_cells) return;
    int x = thread_id % width;
    int y = thread_id / width;
    unsigned char cell = grid_in[thread_id];
    int base_idx = thread_id * MAX_CANDIDATES_PER_CELL;
    int count = 0;

    switch(cell){
        case P_SAND: 
            generate_sand_proposals(grid_in, &proposals[base_idx], &count, x, y, generation, width, height); 
            break;
        case P_WATER: 
            generate_water_proposals(grid_in, &proposals[base_idx], &count, x, y, generation, width, height); 
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

    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    if(tid_x >= width || tid_y >= height) return;
    int thread_id = tid_y * width + tid_x;
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
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    if(tid_x >= width || tid_y >= height) return;
    int thread_id = tid_y * width + tid_x;
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
__global__ void mark_swap_destinations(CellState* cell_states, unsigned char* swap_buffer, int width, int height){
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    if(tid_x >= width || tid_y >= height) return;
    int dest_idx = tid_y * width + tid_x;
    
    CellState* dest_state = &cell_states[dest_idx];
    
    // Rimuovi questa riga: swap_buffer[dest_idx] = 0;
    
    // Se qualcuno ha vinto uno swap verso questa destinazione
    if(dest_state->resolved && dest_state->winner_is_swap){
        int src_x = dest_state->winner_src_x;
        int src_y = dest_state->winner_src_y;
        int src_idx = src_y * width + src_x;
        
        swap_buffer[src_idx] = 1;  // Usa 1 invece di 255 per chiarezza
    }
}
__global__ void apply_movements(CellState* cell_states, unsigned char* grid_out, unsigned char* grid_in, unsigned char* swap_buffer, int width, int height){
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    if(tid_x >= width || tid_y >= height) return;
    int thread_id = tid_y * width + tid_x;
    
    CellState* cell_state = &cell_states[thread_id];
    unsigned char original = grid_in[thread_id];
    
    // Se questa cella è la SORGENTE di uno swap, diventerà VUOTA (o acqua)
    // NON manteniamo l'originale!
    if(swap_buffer[thread_id] == 1){
        grid_out[thread_id] = P_EMPTY;  // ← CORREZIONE: svuota la sorgente
        return;
    }
    
    // Se questa cella è resolved
    if(cell_state->resolved){
        // Movimento normale o destinazione di swap
        grid_out[thread_id] = cell_state->final_type;
    } else {
        // Nessun movimento
        grid_out[thread_id] = original;
    }
}
__global__ void complete_swaps(CellState* cell_states, unsigned char* grid_out, unsigned char* grid_in, unsigned char* swap_buffer, int width, int height){
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    if(tid_x >= width || tid_y >= height) return;
    int src_idx = tid_y * width + tid_x;
    
    // Se questa posizione è la sorgente di uno swap
    if(swap_buffer[src_idx] == 1){
        unsigned char original_at_src = grid_in[src_idx];
        
        // Trova la destinazione (deve essere sotto)
        int below_idx = (tid_y + 1) * width + tid_x;
        if(tid_y + 1 < height){
            CellState* below_state = &cell_states[below_idx];
            if(below_state->resolved && below_state->winner_is_swap &&
               below_state->winner_src_x == tid_x && below_state->winner_src_y == tid_y){
                
                unsigned char original_at_dest = grid_in[below_idx];
                
                // Verifica che sia un vero swap SAND-WATER
                if(original_at_src == P_SAND && original_at_dest == P_WATER){
                    // L'acqua sale nella posizione della sabbia
                    grid_out[src_idx] = P_WATER;
                } else {
                    // Sicurezza: se non è uno swap valido, mantieni vuoto
                    grid_out[src_idx] = P_EMPTY;
                }
            }
        }
    }
}