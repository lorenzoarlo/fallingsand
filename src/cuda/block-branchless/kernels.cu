#include "kernels.cuh"

__device__ inline void ifSwap(bool condition, unsigned char *a, unsigned char *b)
{
    unsigned char mask = -condition; // 0xFF se true, 0x00 se false
    unsigned char temp = (*a ^ *b) & mask;
    *a = *a ^ temp;
    *b = *b ^ temp;
}

__device__ inline float rh(unsigned int base, unsigned int seed)
{
    unsigned int n = base ^ (seed * 387413);
    n = (n ^ (n >> 13)) * 1274126177;
    return (float)(n & 0xFFFF) / 65535.0f;
}

__device__ inline unsigned char get_cell(unsigned char *grid, int x, int y, int width, int height) {
    if (x < 0 || y < 0 || x >= width || y >= height) return P_WALL;
    return grid[y * width + x];
}

/**
 * Function that calculates the new state of the particle
 */
__device__ inline unsigned char calculate(
    unsigned char topleft, unsigned char topright,
    unsigned char bottomleft, unsigned char bottomright,
    bool under_floor_solid,
    int generation, int width, int height,
    int x, int y,
    unsigned char *grid_in,
    unsigned char *grid_out)
{
    unsigned int base = (x * 374761393) ^ (y * 668265263) ^ (generation * 1274126177);

    float r0 = rh(base, 0);
    float r1 = rh(base, 1);
    float r2 = rh(base, 2);
    float r3 = rh(base, 3);
    float r4 = rh(base, 4);
    float r5 = rh(base, 5);

    // --- SAND movement ---
    int topsand_can_move = (topleft == P_SAND && topright < P_SAND) ||
                           (topright == P_SAND && topleft < P_SAND);
    int sand_is_falling = bottomleft < P_SAND && bottomright < P_SAND;

    bool top_sand_moved_hor = (topsand_can_move && sand_is_falling && r0 < SAND_NOISE_CHANCE);
    ifSwap(top_sand_moved_hor, &topleft, &topright);

    // TOPLEFT
    bool topleft_can_fall_vertical = (topleft == P_SAND && bottomleft < P_SAND && r1 < SAND_FALL_DOWN_CHANCE);
    bool topleft_diagonally = (topleft == P_SAND && bottomleft >= P_SAND && topright < P_SAND && bottomright < P_SAND);

    ifSwap((!top_sand_moved_hor && topleft_can_fall_vertical), &topleft, &bottomleft);
    ifSwap((!top_sand_moved_hor && topleft_diagonally), &topleft, &bottomright);

    // TOPRIGHT
    bool topright_can_fall_vertical = (topright == P_SAND && bottomright < P_SAND && r1 < SAND_FALL_DOWN_CHANCE);
    bool topright_diagonally = (topright == P_SAND && bottomleft < P_SAND && topleft < P_SAND && bottomright >= P_SAND);

    ifSwap((!top_sand_moved_hor && topright_can_fall_vertical), &topright, &bottomright);
    ifSwap((!top_sand_moved_hor && topright_diagonally), &topright, &bottomleft);

    // TOPLEFT WATER PARTICLE
    bool watertopleft_can_go_down = topleft == P_WATER  && bottomleft < topleft && r2 < WATER_FALL_DENSITY_CHANCE;
    bool watertopleft_can_go_diag = topleft == P_WATER  && !watertopleft_can_go_down && topright < topleft && bottomright < topleft && r3 < WATER_MOVE_DIAGONAL_CHANCE;

    ifSwap(watertopleft_can_go_down, &topleft, &bottomleft);
    ifSwap(watertopleft_can_go_diag, &topleft, &bottomright);

    // TOPRIGHT WATER PARTICLE
    bool watertopright_can_go_down = topright == P_WATER && bottomright < topright && r2 < WATER_FALL_DENSITY_CHANCE;
    bool watertopright_can_go_diag = topright == P_WATER && !watertopright_can_go_down && topleft < topright && bottomleft < topright && r3 < WATER_MOVE_DIAGONAL_CHANCE;

    ifSwap(watertopright_can_go_down, &topright, &bottomright);
    ifSwap(watertopright_can_go_diag, &topright, &bottomleft);

    bool water_dropped = watertopleft_can_go_down || watertopleft_can_go_diag || watertopright_can_go_down || watertopright_can_go_diag;

    // Horizontal movement for WATER particles if they didn't drop

    bool top_can_move_horizontally = (topleft == P_WATER && topright < P_WATER) ||
                                     (topleft < P_WATER && topright == P_WATER);
    bool below_solid = bottomleft >= P_WATER && bottomright >= P_WATER;

    ifSwap((!water_dropped && top_can_move_horizontally && (below_solid || r4 < WATER_MOVE_HORIZONTAL_CHANCE)), &topleft, &topright);

    // Bottom row horizontal movement

    int bottom_can_move_horizontally = (bottomleft == P_WATER && bottomright < P_WATER) ||
                                       (bottomleft < P_WATER && bottomright == P_WATER);

    ifSwap((bottom_can_move_horizontally && (under_floor_solid || r5 < WATER_MOVE_HORIZONTAL_CHANCE)), &bottomleft, &bottomright);

    int i_topleft       = (y * width) + x;
    int i_topright      = (y * width) + (x + 1);
    int i_bottomleft    = ((y + 1) * width) + x;
    int i_bottomright   = ((y + 1) * width) + (x + 1);

    grid_out[i_topleft]         = topleft;
    grid_out[i_topright]        = topright;
    grid_out[i_bottomleft]      = bottomleft;
    grid_out[i_bottomright]     = bottomright;
}

__global__ void kernel_opt(unsigned char* grid_in, unsigned char* grid_out,
                                  int width, int height, int generation)
{
    int phase = generation % 4;
    int offset_x = (phase == 1 || phase == 3);
    int offset_y = (phase == 1 || phase == 2);

    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

    int x = tid_x * 2 + offset_x;
    int y = tid_y * 2 + offset_y;

    if(x >= width-1 || y >= height-1) return;


    int i_topleft = (y * width) + x;
    int i_topright = (y * width) + (x + 1);
    int i_bottomleft = ((y + 1) * width) + x;
    int i_bottomright = ((y + 1) * width) + (x + 1);
    // Pointers to the 4 cells
    unsigned char topleft = grid_in[i_topleft];
    unsigned char topright = grid_in[i_topright];
    unsigned char bottomleft = grid_in[i_bottomleft];
    unsigned char bottomright = grid_in[i_bottomright];

    bool under_floor_solid = get_cell(grid_in, x, y + 2, width, height) >= P_WATER && get_cell(grid_in, x + 1, y + 2, width, height) >= P_WATER;

    if(!(topleft == topright && bottomleft == bottomright && topleft == bottomleft))
        calculate(topleft, topright, bottomleft, bottomright, under_floor_solid,
                  generation, width, height, x, y, grid_in, grid_out);
}
