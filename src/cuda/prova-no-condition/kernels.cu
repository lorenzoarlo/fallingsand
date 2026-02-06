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

__device__ inline unsigned char get_cell(unsigned char *grid, int width, int height, int x, int y) {
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
    int anchor_x, int anchor_y,
    unsigned char *grid_in,
    unsigned char *grid_out)
{
    unsigned int base = (anchor_x * 374761393) ^ (anchor_y * 668265263) ^ (generation * 1274126177);

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
    bool topleft_can_fall_vertical = (topleft == P_SAND && bottomleft < P_SAND && r1 < WATER_FALL_DOWN_CHANCE);
    bool topleft_diagonally = (topleft == P_SAND && bottomleft >= P_SAND && topright < P_SAND && bottomright < P_SAND);

    ifSwap((!top_sand_moved_hor && topleft_can_fall_vertical), &topleft, &bottomleft);
    ifSwap((!top_sand_moved_hor && topleft_diagonally), &topleft, &bottomright);

    // TOPRIGHT
    bool topright_can_fall_vertical = (topright == P_SAND && bottomright < P_SAND && r1 < WATER_FALL_DOWN_CHANCE);
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

    bool water_dropped = watertopleft_can_go_down | watertopleft_can_go_diag | watertopright_can_go_down | watertopright_can_go_diag;

    // Horizontal movement for WATER particles if they didn't drop

    bool top_can_move_horizontally = (topleft == P_WATER && topright < P_WATER) ||
                                     (topleft < P_WATER && topright == P_WATER);
    bool below_solid = bottomleft >= P_WATER && bottomright >= P_WATER;

    ifSwap((!water_dropped && top_can_move_horizontally && (below_solid || r4 < WATER_MOVE_HORIZONTAL_CHANCE)), &topleft, &topright);

    // Bottom row horizontal movement

    int bottom_can_move_horizontally = (bottomleft == P_WATER && bottomright < P_WATER) ||
                                       (bottomleft < P_WATER && bottomright == P_WATER);

    ifSwap((bottom_can_move_horizontally && (under_floor_solid || r5 < WATER_MOVE_HORIZONTAL_CHANCE)), &bottomleft, &bottomright);

    int mask_x = -x;          // Mask for x
    int not_mask_x = ~mask_x; // Mask for NOT x

    int mask_y = -y;          // Mask for y
    int not_mask_y = ~mask_y; // Mask for NOT y
    
    // Lookup table for the 2x2 block, using bitwise operations to select the correct value based on local_x and local_y
    return (topleft & not_mask_x & not_mask_y) |
           (topright & mask_x & not_mask_y) |
           (bottomleft & not_mask_x & mask_y) |
           (bottomright & mask_x & mask_y);
}

// make default_test_cuda FRAMES=1500 SCALE=1 SAMPLE=2 LOGIC=src/cuda/prova-no-condition
__global__ void kernel(unsigned char *grid_in, unsigned char *grid_out, int width, int height, int generation)
{
    int phase = generation % 4;
    int offset_x = (phase == 1 || phase == 3);
    int offset_y = (phase == 1 || phase == 2);

    // Absolute coordinates
    int x = (blockIdx.x * blockDim.x + threadIdx.x);
    int y = (blockIdx.y * blockDim.y + threadIdx.y);

    // Shifting coordinates according to the phase
    int shifted_x = x - offset_x;
    int shifted_y = y - offset_y;

    // Check bounds
    if (shifted_x < 0 || shifted_y < 0 || x >= width || y >= height)
        return;

    // local positions within the 2x2 block
    int local_x = shifted_x % 2; // 0 = left, 1 = right
    int local_y = shifted_y % 2; // 0 = top, 1 = bottom

    // Coordinates of topleft of the 2x2 block
    /*int anchor_x = x - local_x;
    int anchor_y = y - local_y;*/

    int anchor_x = ((x - offset_x) / 2) * 2 + offset_x;
    int anchor_y = ((y - offset_y) / 2) * 2 + offset_y;

    // Check bounds for the 2x2 block
    if (anchor_x + 1 >= width || anchor_y + 1 >= height) 
        return;

    local_x = x - anchor_x;
    local_y = y - anchor_y;

    // Calculates cell values
    unsigned char topleft = grid_in[anchor_y * width + anchor_x];
    unsigned char topright = grid_in[anchor_y * width + anchor_x + 1];
    unsigned char bottomleft = grid_in[(anchor_y + 1) * width + anchor_x];
    unsigned char bottomright = grid_in[(anchor_y + 1) * width + anchor_x + 1];

    bool under_floor_solid = (anchor_y + 2 >= height || (grid_in[(anchor_y + 2) * width + anchor_x] >= P_WATER && grid_in[(anchor_y + 2) * width + anchor_x + 1] >= P_WATER));

    /*if (!(topleft == topright && bottomleft == bottomright && topleft == bottomleft))
    {*/
        grid_out[y * width + x] = calculate(topleft, topright, bottomleft, bottomright, under_floor_solid,
                                            generation, width, height, local_x, local_y, anchor_x, anchor_y, grid_in, grid_out);
    //}
}
