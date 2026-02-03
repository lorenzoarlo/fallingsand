#include "kernels.cuh"

__device__ float random_hash(int x, int y, int generation, int seed){
    unsigned int n = (x * 374761393) ^ (y * 668265263) ^ (generation * 1274126177) ^ (seed * 387413);
    n = (n ^ (n >> 13)) * 1274126177;
    return (float)(n & 0xFFFF) / 65535.0f;
}

__device__ void swap(unsigned char *a, unsigned char *b){
    unsigned char temp = *a;
    *a = *b;
    *b = temp;
}

__device__ void set_cell(unsigned char* cell, unsigned char value){
    *cell = value;
}

__device__ unsigned char get_cell(unsigned char* grid, int x, int y, int width, int height){
    if(x < 0 || x >= width || y < 0 || y >= height)
        return P_WALL;
    return grid[y * width + x];
}

__global__ void kernel(unsigned char* grid_in, unsigned char* grid_out, int width, int height, int offset_x, int offset_y, int generation){
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
            unsigned char *topleft = &grid_out[i_topleft];
            unsigned char *topright = &grid_out[i_topright];
            unsigned char *bottomleft = &grid_out[i_bottomleft];
            unsigned char *bottomright = &grid_out[i_bottomright];

            // If all cells are the same, skip processing
            if (*topleft == *topright && *bottomleft == *bottomright && *topleft == *bottomleft)
                return;

            int topleft_moved = 0;
            int topright_moved = 0;

            // Manage horizontal SAND movement (to make it more noisy when falling)
            int topsand_can_move = *topleft == P_SAND && *topright < P_SAND ||
                                   *topright == P_SAND && *topleft < P_SAND;

            int sand_is_falling = *bottomleft < P_SAND && *bottomright < P_SAND;
            if (topsand_can_move && sand_is_falling && random_hash(x, y, generation, 0) < SAND_NOISE_CHANCE)
            {
                swap(topleft, topright);
                topleft_moved = 1;
                topright_moved = 1;
            }

            // Manage SAND falling

            // topleft sand particle
            if (!topleft_moved && *topleft == P_SAND)
            {
                // Can fall down?
                if (*bottomleft < P_SAND)
                {
                    if (random_hash(x, y, generation, 1) < WATER_FALL_DOWN_CHANCE)
                    {
                        swap(topleft, bottomleft);
                    }
                }
                // Can fall diagonally?
                else if (*topright < P_SAND && *bottomright < P_SAND)
                {
                    swap(topleft, bottomright);
                }
            }
            // topright sand particle
            if (!topright_moved && *topright == P_SAND)
            {
                // Can fall down
                if (*bottomright < P_SAND)
                {
                    if (random_hash(x, y, generation, 1) < WATER_FALL_DOWN_CHANCE)
                    {
                        swap(topright, bottomright);
                    }
                }
                else if (*topleft < P_SAND && *bottomleft < P_SAND)
                {
                    swap(topright, bottomleft); // Cade in diagonale
                }
            }

            // Manage WATER falling and horizontal movement
            int drop_left = 0;
            int drop_right = 0;

            // topleft water particle
            if (*topleft == P_WATER)
            {
                // Is the cell below lower density?
                if (*bottomleft < *topleft && random_hash(x, y, generation, 2) < WATER_FALL_DENSITY_CHANCE)
                {
                    swap(topleft, bottomleft);
                    drop_left = 1;
                }
                // Can move diagonally?
                else if (*topright < *topleft && *bottomright < *topleft && random_hash(x, y, generation, 3) < WATER_MOVE_DIAGONAL_CHANCE)
                {
                    swap(topleft, bottomright);
                    drop_left = 1;
                }
            }

            // topright water particle
            if (*topright == P_WATER)
            {
                // Is the cell below lower density?
                if (*bottomright < *topright && random_hash(x, y, generation, 2) < WATER_FALL_DENSITY_CHANCE)
                {
                    swap(topright, bottomright);
                    drop_right = 1;
                } // Can move diagonally?
                else if (*topleft < *topright && *bottomleft < *topright && random_hash(x, y, generation, 3) < WATER_MOVE_DIAGONAL_CHANCE)
                {
                    swap(topright, bottomleft);
                    drop_right = 1;
                }
            }

            // Horizontal water movement if not dropped
            if (!drop_left && !drop_right)
            {
                int top_can_move_horizontally = (*topleft == P_WATER && *topright < P_WATER ||
                                                 *topleft < P_WATER && *topright == P_WATER);
                int below_solid = *bottomleft >= P_WATER && *bottomright >= P_WATER;

                if (top_can_move_horizontally && (below_solid || random_hash(x, y, generation, 4) < WATER_MOVE_HORIZONTAL_CHANCE))
                {
                    swap(topleft, topright);
                }
            }

            int bottom_can_move_horizontally = (*bottomleft == P_WATER && *bottomright < P_WATER ||
                                                *bottomleft < P_WATER && *bottomright == P_WATER);
            // Look if there is solid floor below
            int floor_is_solid = get_cell(grid_in, x, y + 2, width, height) >= P_WATER && get_cell(grid_in, x + 1, y + 2, width, height) >= P_WATER;

            // Swap of below cells if possible
            // Use different random channel to avoid correlation with top movement
            if (bottom_can_move_horizontally && (floor_is_solid || random_hash(x, y, generation, 5) < WATER_MOVE_HORIZONTAL_CHANCE))
            {
                swap(bottomleft, bottomright);
            }
}