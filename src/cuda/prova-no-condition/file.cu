/**
 *  Logic to process a 2x2 block using branchless scalar code
 

#include "../../simulation.h"
#include "kernels.cuh"

__device__ inline float random_hash(int x, int y, int generation, int seed){
    unsigned int n = (x * 374761393) ^ (y * 668265263) ^ (generation * 1274126177) ^ (seed * 387413);
    n = (n ^ (n >> 13)) * 1274126177;
    return (float)(n & 0xFFFF) / 65535.0f;
}

__device__ inline unsigned char get_cell(Universe *u, int x, int y){
    if(x < 0 || x >= u->width || y < 0 || y >= u->height)
        return P_WALL;
    return u->cells[y * u->width + x];
}

static inline void blocklogic(int x, int y, int width, unsigned char *cells, int generation, Universe *u)
{
    // calculate index of the 4 cells
    int i_topleft = (y * width) + x;
    int i_topright = (y * width) + (x + 1);
    int i_bottomleft = ((y + 1) * width) + x;
    int i_bottomright = ((y + 1) * width) + (x + 1);

    // Pointers to the 4 cells
    unsigned char *topleft = &cells[i_topleft];
    unsigned char *topright = &cells[i_topright];
    unsigned char *bottomleft = &cells[i_bottomleft];
    unsigned char *bottomright = &cells[i_bottomright];

    // If all cells are the same, skip processing
    if (*topleft == *topright && *bottomleft == *bottomright && *topleft == *bottomleft)
    {
        return;
    }

    // Manage horizontal SAND movement (to make it more noisy when falling)
    int topsand_can_move = (*topleft == P_SAND && *topright < P_SAND) ||
                           (*topright == P_SAND && *topleft < P_SAND);
    int sand_is_falling = (*bottomleft < P_SAND) && (*bottomright < P_SAND);
    int sand_noise_ok = random_hash(x, y, generation, 0) < SAND_NOISE_CHANCE;
    
    int top_moved = topsand_can_move && sand_is_falling && sand_noise_ok;
    COND_SWAP(top_moved, topleft, topright);

    // Manage SAND falling

    // topleft sand particle
    int topleft_is_sand = (*topleft == P_SAND);
    int bottomleft_less_sand = (*bottomleft < P_SAND);
    int topright_less_sand = (*topright < P_SAND);
    int bottomright_less_sand = (*bottomright < P_SAND);
    int water_fall_ok_1 = random_hash(x, y, generation, 1) < WATER_FALL_DOWN_CHANCE;

    int topleft_fall_down = !top_moved && topleft_is_sand && bottomleft_less_sand && water_fall_ok_1;
    COND_SWAP(topleft_fall_down, topleft, bottomleft);

    // Recalculate after potential swap
    topright_less_sand = (*topright < P_SAND);
    bottomright_less_sand = (*bottomright < P_SAND);
    topleft_is_sand = (*topleft == P_SAND);

    int topleft_fall_diag = !top_moved && !topleft_fall_down && topleft_is_sand && 
                            topright_less_sand && bottomright_less_sand;
    COND_SWAP(topleft_fall_diag, topleft, bottomright);

    // topright sand particle - recalculate states
    int topright_is_sand = (*topright == P_SAND);
    bottomright_less_sand = (*bottomright < P_SAND);
    int topleft_less_sand = (*topleft < P_SAND);
    bottomleft_less_sand = (*bottomleft < P_SAND);

    int topright_fall_down = !top_moved && topright_is_sand && bottomright_less_sand && water_fall_ok_1;
    COND_SWAP(topright_fall_down, topright, bottomright);

    // Recalculate for diagonal
    topleft_less_sand = (*topleft < P_SAND);
    bottomleft_less_sand = (*bottomleft < P_SAND);
    topright_is_sand = (*topright == P_SAND);

    int topright_fall_diag = !top_moved && !topright_fall_down && topright_is_sand && 
                             topleft_less_sand && bottomleft_less_sand;
    COND_SWAP(topright_fall_diag, topright, bottomleft);

    // Manage WATER falling and horizontal movement
    int water_dropped = 0;

    // topleft water particle
    int topleft_is_water = (*topleft == P_WATER);
    int bottomleft_less_topleft = (*bottomleft < *topleft);
    int topright_less_topleft = (*topright < *topleft);
    int bottomright_less_topleft = (*bottomright < *topleft);
    int water_fall_density_ok = random_hash(x, y, generation, 2) < WATER_FALL_DENSITY_CHANCE;
    int water_diag_ok = random_hash(x, y, generation, 3) < WATER_MOVE_DIAGONAL_CHANCE;

    int topleft_water_fall = topleft_is_water && bottomleft_less_topleft && water_fall_density_ok;
    COND_SWAP(topleft_water_fall, topleft, bottomleft);
    water_dropped |= topleft_water_fall;

    // Recalculate for diagonal
    topright_less_topleft = (*topright < *topleft);
    bottomright_less_topleft = (*bottomright < *topleft);
    topleft_is_water = (*topleft == P_WATER);

    int topleft_water_diag = !topleft_water_fall && topleft_is_water && 
                             topright_less_topleft && bottomright_less_topleft && water_diag_ok;
    COND_SWAP(topleft_water_diag, topleft, bottomright);
    water_dropped |= topleft_water_diag;

    // topright water particle
    int topright_is_water = (*topright == P_WATER);
    int bottomright_less_topright = (*bottomright < *topright);
    int topleft_less_topright = (*topleft < *topright);
    int bottomleft_less_topright = (*bottomleft < *topright);

    int topright_water_fall = topright_is_water && bottomright_less_topright && water_fall_density_ok;
    COND_SWAP(topright_water_fall, topright, bottomright);
    water_dropped |= topright_water_fall;

    // Recalculate for diagonal
    topleft_less_topright = (*topleft < *topright);
    bottomleft_less_topright = (*bottomleft < *topright);
    topright_is_water = (*topright == P_WATER);

    int topright_water_diag = !topright_water_fall && topright_is_water && 
                              topleft_less_topright && bottomleft_less_topright && water_diag_ok;
    COND_SWAP(topright_water_diag, topright, bottomleft);
    water_dropped |= topright_water_diag;

    // Horizontal water movement if not dropped
    int top_can_move_horizontally = (*topleft == P_WATER && *topright < P_WATER) ||
                                    (*topleft < P_WATER && *topright == P_WATER);
    int below_solid = (*bottomleft >= P_WATER) && (*bottomright >= P_WATER);
    int water_horiz_ok = random_hash(x, y, generation, 4) < WATER_MOVE_HORIZONTAL_CHANCE;

    int top_water_swap = !water_dropped && top_can_move_horizontally && (below_solid || water_horiz_ok);
    COND_SWAP(top_water_swap, topleft, topright);

    // Bottom water horizontal movement
    int bottomwater_can_move_horizontally = (*bottomleft == P_WATER && *bottomright < P_WATER) ||
                                            (*bottomleft < P_WATER && *bottomright == P_WATER);
    int floor_is_solid = (get_cell(u, x, y + 2) >= P_WATER) && (get_cell(u, x + 1, y + 2) >= P_WATER);
    int water_horiz_ok_5 = random_hash(x, y, generation, 5) < WATER_MOVE_HORIZONTAL_CHANCE;

    int bottom_water_swap = bottomwater_can_move_horizontally && (floor_is_solid || water_horiz_ok_5);
    COND_SWAP(bottom_water_swap, bottomleft, bottomright);

    #undef COND_SWAP
}*/