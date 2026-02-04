#include "kernels.cuh"

__device__ inline void calculate(
    unsigned char topleft, unsigned char topright,
    unsigned char bottomleft, unsigned char bottomright,
    int generation, int width, int height,
    int x, int y,
    unsigned char* grid_in,
    unsigned char* grid_out
){
    unsigned int base = (x * 374761393) ^ (y * 668265263) ^ (generation * 1274126177);
    auto rh = [&](unsigned int seed){
        unsigned int n = base ^ (seed * 387413);
        n = (n ^ (n >> 13)) * 1274126177;
        return (float)(n & 0xFFFF) / 65535.0f;
    };

    float r0 = rh(0);
    float r1 = rh(1);
    float r2 = rh(2);
    float r3 = rh(3);
    float r4 = rh(4);
    float r5 = rh(5);

    unsigned char topleft_moved = 0;
    unsigned char topright_moved = 0;
    unsigned char temp;

    // --- Horizontal SAND movement ---
    int topsand_can_move = (topleft == P_SAND && topright < P_SAND) ||
                           (topright == P_SAND && topleft < P_SAND);
    int sand_is_falling = bottomleft < P_SAND && bottomright < P_SAND;
    unsigned char condition = (topsand_can_move && sand_is_falling && r0 < SAND_NOISE_CHANCE);

    unsigned char a = topleft;
    unsigned char b = topright;
    topleft  = condition ? b : a;
    topright = condition ? a : b;
    topleft_moved |= condition;
    topright_moved |= condition;

    // --- Sand falling ---
    if (!topleft_moved && topleft == P_SAND)
    {
        if (bottomleft < P_SAND && r1 < WATER_FALL_DOWN_CHANCE)
        {
            temp = topleft; topleft = bottomleft; bottomleft = temp;
        }
        else if (topright < P_SAND && bottomright < P_SAND)
        {
            temp = topleft; topleft = bottomright; bottomright = temp;
        }
    }
    if (!topright_moved && topright == P_SAND)
    {
        if (bottomright < P_SAND && r1 < WATER_FALL_DOWN_CHANCE)
        {
            temp = topright; topright = bottomright; bottomright = temp;
        }
        else if (topleft < P_SAND && bottomleft < P_SAND)
        {
            temp = topright; topright = bottomleft; bottomleft = temp;
        }
    }

    // --- Water falling and horizontal movement ---
    unsigned char drop_left = 0;
    unsigned char drop_right = 0;

    // TOP LEFT
    bool iswater = topleft == P_WATER;
    bool can_go_down = iswater && bottomleft < topleft && r2 < WATER_FALL_DENSITY_CHANCE;
    bool can_go_diag = iswater && !can_go_down && topright < topleft && bottomright < topleft && r3 < WATER_MOVE_DIAGONAL_CHANCE;

    a = topleft; b = bottomleft;
    topleft = can_go_down ? b : a; bottomleft = can_go_down ? a : b;

    a = topleft; b = bottomright;
    topleft = can_go_diag ? b : a; bottomright = can_go_diag ? a : b;

    drop_left |= (can_go_down | can_go_diag);

    // TOP RIGHT
    iswater = topright == P_WATER;
    can_go_down = iswater && bottomright < topright && r2 < WATER_FALL_DENSITY_CHANCE;
    can_go_diag = iswater && !can_go_down && topleft < topright && bottomleft < topright && r3 < WATER_MOVE_DIAGONAL_CHANCE;

    a = topright; b = bottomright;
    topright = can_go_down ? b : a; bottomright = can_go_down ? a : b;

    a = topright; b = bottomleft;
    topright = can_go_diag ? b : a; bottomleft = can_go_diag ? a : b;

    drop_right |= (can_go_down | can_go_diag);

    // Horizontal water movement if not dropped
    if (!drop_left && !drop_right)
    {
        int top_can_move_horizontally = (topleft == P_WATER && topright < P_WATER) ||
                                        (topleft < P_WATER && topright == P_WATER);
        int below_solid = bottomleft >= P_WATER && bottomright >= P_WATER;
        condition = (top_can_move_horizontally && (below_solid || r4 < WATER_MOVE_HORIZONTAL_CHANCE));
        a = topleft; b = topright;
        topleft = condition ? b : a; topright = condition ? a : b;
    }

    int bottom_can_move_horizontally = (bottomleft == P_WATER && bottomright < P_WATER) ||
                                       (bottomleft < P_WATER && bottomright == P_WATER);
    // Read floor directly from grid_in
    unsigned int yp2 = y + 2;
    unsigned char first = (x >= width || yp2 >= height) ? P_WALL : grid_in[yp2*width + x];
    unsigned char second = (x+1 >= width || yp2 >= height) ? P_WALL : grid_in[yp2*width + (x+1)];
    int floor_is_solid = first >= P_WATER && second >= P_WATER;

    if (bottom_can_move_horizontally && (floor_is_solid || r5 < WATER_MOVE_HORIZONTAL_CHANCE))
    {
        temp = bottomleft; bottomleft = bottomright; bottomright = temp;
    }

    // Write results
    int idx = y*width + x;
    grid_out[idx]         = topleft;
    grid_out[idx + 1]     = topright;
    grid_out[idx + width] = bottomleft;
    grid_out[idx + width+1] = bottomright;
}

__global__ void kernel(unsigned char* grid_in, unsigned char* grid_out, int width, int height, int generation)
{
    int phase = generation % 4;
    int offset_x = (phase == 1 || phase == 3);
    int offset_y = (phase == 1 || phase == 2);

    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

    int x = (tid_x << 1) + offset_x;
    int y = (tid_y << 1) + offset_y;

    if(x >= width-1 || y >= height-1) return;

    // Shared memory per blocco
    __shared__ unsigned char tile[BLOCK_SIZE*2+4][BLOCK_SIZE*2+4];

    int lx = threadIdx.x*2 + 2;
    int ly = threadIdx.y*2 + 2;

    // Copia le celle centrali del thread
    for(int dy=0; dy<2; dy++){
        for(int dx=0; dx<2; dx++){
            int gx = x + dx;
            int gy = y + dy;
            tile[ly+dy][lx+dx] = (gx < width && gy < height) ? grid_in[gy*width + gx] : P_WALL;
        }
    }

    __syncthreads();

    unsigned char topleft     = tile[ly][lx];
    unsigned char topright    = tile[ly][lx+1];
    unsigned char bottomleft  = tile[ly+1][lx];
    unsigned char bottomright = tile[ly+1][lx+1];

    if(topleft == topright && bottomleft == bottomright && topleft == bottomleft) return;

    calculate(topleft, topright, bottomleft, bottomright, generation, width, height, x, y, grid_in, grid_out);
}
