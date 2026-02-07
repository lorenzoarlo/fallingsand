#include "kernels.cuh"

__global__ void kernel(unsigned char* grid_in, unsigned char* grid_out, int width, int height, int offset_x, int offset_y, int generation){


    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

    int x = (tid_x << 1) + offset_x;
    int y = (tid_y << 1) + offset_y;

    if(x >= width-1 || y >= height-1) return;

    unsigned char temp;

    int i_topleft = (y * width) + x;
    int i_topright = (y * width) + (x + 1);
    int i_bottomleft = ((y + 1) * width) + x;
    int i_bottomright = ((y + 1) * width) + (x + 1);
    
    unsigned char topleft = grid_out[i_topleft];
    unsigned char topright = grid_out[i_topright];
    unsigned char bottomleft = grid_out[i_bottomleft];
    unsigned char bottomright = grid_out[i_bottomright];
    // If all cells are the same, skip processing
    if (topleft == topright && bottomleft == bottomright && topleft == bottomleft)
        return;

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
    
    // Manage horizontal SAND movement (to make it more noisy when falling)
    int topsand_can_move = (topleft == P_SAND && topright < P_SAND) ||
                           (topright == P_SAND && topleft < P_SAND);
    int sand_is_falling = bottomleft < P_SAND && bottomright < P_SAND;

    unsigned char condition = (topsand_can_move && sand_is_falling && r0 < SAND_NOISE_CHANCE);
    unsigned char a = topleft;
    unsigned char b = topright;

    topleft = condition ? b : a;
    topright = condition ? a : b;
    // OR se condition è 0 lascia il valore di prima
    topleft_moved |= condition;
    topright_moved |= condition;

    // Manage SAND falling
    // topleft sand particle
    if (!topleft_moved && topleft == P_SAND)
    {
        // Can fall down?
        if (bottomleft < P_SAND)
        {
            if (r1 < SAND_FALL_DOWN_CHANCE)
            {
                temp = topleft;
                topleft = bottomleft;
                bottomleft = temp;
            }
        }
        // Can fall diagonally?
        else if (topright < P_SAND && bottomright < P_SAND)
        {
            temp = topleft;
            topleft = bottomright;
            bottomright = temp;
        }
    }
    // topright sand particle
    if (!topright_moved && topright == P_SAND)
    {
        // Can fall down
        if (bottomright < P_SAND)
        {
            if (r1 < SAND_FALL_DOWN_CHANCE)
            {
                temp = topright;
                topright = bottomright;
                bottomright = temp;
            }
        }
        else if (topleft < P_SAND && bottomleft < P_SAND)
        {
            temp = topright;
            topright = bottomleft;
            bottomleft = temp; // Cade in diagonale
        }
    }
    // Manage WATER falling and horizontal movement
    unsigned char drop_left = 0;
    unsigned char drop_right = 0;

    // WATER: TOP LEFT
    bool iswater = topleft == P_WATER;
    bool can_go_down = iswater && bottomleft < topleft && r2 < WATER_FALL_DENSITY_CHANCE;
    bool can_go_diag = iswater && !can_go_down && topright < topleft && bottomright < topleft && r3 < WATER_MOVE_DIAGONAL_CHANCE;

    a = topleft;
    b = bottomleft;
    topleft = can_go_down ? b : a;
    bottomleft = can_go_down ? a : b;

    a = topleft;
    b = bottomright;
    topleft = can_go_diag ? b : a;
    bottomright = can_go_diag ? a : b;

    drop_left |= (can_go_down | can_go_diag);

    // WATER: TOP RIGHT
    iswater = topright == P_WATER;
    can_go_down = iswater && bottomright < topright && r2 < WATER_FALL_DENSITY_CHANCE;
    can_go_diag = iswater && !can_go_down && topleft < topright && bottomleft < topright && r3 < WATER_MOVE_DIAGONAL_CHANCE;

    a = topright;
    b = bottomright; 
    topright = can_go_down ? b : a;
    bottomright = can_go_down ? a : b;

    a = topright;
    b = bottomleft;
    topright = can_go_diag ? b : a;
    bottomleft = can_go_diag ? a : b;

    drop_right |= (can_go_down | can_go_diag);

    // Horizontal water movement if not dropped
    if (!drop_left && !drop_right)
    {
        int top_can_move_horizontally = (topleft == P_WATER && topright < P_WATER ||
                                         topleft < P_WATER && topright == P_WATER);
        int below_solid = bottomleft >= P_WATER && bottomright >= P_WATER;
        condition = (top_can_move_horizontally && (below_solid || r4 < WATER_MOVE_HORIZONTAL_CHANCE));
        a = topleft;
        b = topright;
        topleft = condition ? b : a;
        topright = condition ? a : b;
    }
    int bottom_can_move_horizontally = (bottomleft == P_WATER && bottomright < P_WATER ||
                                        bottomleft < P_WATER && bottomright == P_WATER);
    // Look if there is solid floor below

    unsigned int yp2 = y + 2;
    unsigned char first = (x >= width || yp2 >= height) ? P_WALL : grid_in[(yp2) * width + x];
    unsigned char second = (x + 1 >= width || yp2 >= height) ? P_WALL : grid_in[(yp2) * width + (x+1)];
    int floor_is_solid =  first >= P_WATER && 
                            second >= P_WATER;
    // Swap of below cells if possible
    // Use different random channel to avoid correlation with top movement
    if (bottom_can_move_horizontally && (floor_is_solid || r5 < WATER_MOVE_HORIZONTAL_CHANCE))
    {
        temp = bottomleft;
        bottomleft = bottomright;
        bottomright = temp;
    }

    grid_out[i_topleft] = topleft ;
    grid_out[i_topright] = topright ;
    grid_out[i_bottomleft] = bottomleft ;
    grid_out[i_bottomright] = bottomright ;
}