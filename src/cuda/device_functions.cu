#include "cuda_context.cuh"

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

__device__ int device_out_of_bounds(int x, int y, int width, int height){
    return (x < 0 || x >= width || y < 0 || y >= height);
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
    p.swap_with = swap_with;
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
        proposals[*count] = create_proposal(x, y, x, y + 1, P_SAND, (unsigned char) pref, priority, 0, P_EMPTY);
        (*count)++;
    }
    pref++;

    // Pref 1: First diagonal if below was not empty
    unsigned char diag1_below = device_grid_get(grid, diag1_x, y + 1, width, height);
    if(below != P_EMPTY && diag1_below == P_EMPTY){
        proposals[*count] = create_proposal(x, y, diag1_x, y + 1, P_SAND, (unsigned char) pref, priority, 0, P_EMPTY);
        (*count)++;
    }
    pref++;

    // Pref 2: Second diagonal if below was not empty
    unsigned char diag2_below = device_grid_get(grid, diag2_x, y + 1, width, height);
    if(below != P_EMPTY && diag2_below == P_EMPTY){
        proposals[*count] = create_proposal(x, y, diag2_x, y + 1, P_SAND, (unsigned char) pref, priority, 0, P_EMPTY);
        (*count)++;
    }
    pref++;

    // Pref 3: Swap with water below
    if(below == P_WATER){
        if ((x + y + generation) % 2 == 0) {
            proposals[*count] = create_proposal(x, y, x, y + 1, P_SAND, (unsigned char) pref, priority, 1, P_WATER);
            (*count)++;
        } else {
            proposals[*count] = create_proposal(x, y, x, y, P_SAND, (unsigned char) 4, priority, 0, P_EMPTY);
            (*count)++;
            return;  // Don't add fallback, we already have stay-in-place
        }
    }
    pref++;

    // Pref 4: Stay in place
    if(*count == 0 || proposals[*count - 1].dest_x != x || proposals[*count - 1].dest_y != y){
        proposals[*count] = create_proposal(x, y, x, y, P_SAND, (unsigned char) pref, priority, 0, P_EMPTY);
        (*count)++;
    }
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
