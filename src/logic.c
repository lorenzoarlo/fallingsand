#include "simulation.h"
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

#define SAND_NOISE_CHANCE 0.4f
#define WATER_FALL_DOWN_CHANCE 0.9f
#define WATER_FALL_DENSITY_CHANCE 1.0f
#define WATER_MOVE_DIAGONAL_CHANCE 0.5f
#define WATER_MOVE_HORIZONTAL_CHANCE 0.8f

/**
 * Pseudo random function based on hash but deterministic
 */
static inline float random_hash(int x, int y, int frame, int salt)
{
    unsigned int n = (x * 374761393) ^ (y * 668265263) ^ (frame * 1274126177) ^ (salt * 387413);
    n = (n ^ (n >> 13)) * 1274126177;
    return (float)(n & 0xFFFF) / 65535.0f;
}

/**
 *  Utility function to swap two cells
 */
static inline void swap(unsigned char *a, unsigned char *b)
{
    unsigned char temp = *a;
    *a = *b;
    *b = temp;
}

/**
 * Utility function to set cell type
 */
static inline void set_cell(unsigned char *cell, unsigned char type)
{
    *cell = type;
}

/**
 * Utility function to get cell type with out-of-bounds check
 */
static inline unsigned char get_cell(Universe *u, int x, int y)
{
    if (universe_out_of_bounds(u, x, y))
    {
        return P_WALL;
    }
    return u->cells[UINDEX(x, y, u->width)];
}

/**
 * Compute the next generation of the universe
 */
void next(Universe *u, Universe *out, int generation)
{

    // Copying by default the entire universe
    memcpy(out->cells, u->cells, u->width * u->height);

    // Choose which cell of the 2x2 block to offset based on generation
    // Pattern Margolus for alternates (0,0) -> (1,1) -> (0,1) -> (1,0)
    int phase = generation % 4;
    int offset_x = (phase == 1 || phase == 3) ? 1 : 0;
    int offset_y = (phase == 1 || phase == 2) ? 1 : 0;

    // Iterate over the universe in 2x2 blocks
    for (int y = offset_y; y < u->height - 1; y += 2)
    {
        for (int x = offset_x; x < u->width - 1; x += 2)
        {
            // calculate index of the 4 cells
            int i_topleft = (y * u->width) + x;
            int i_topright = (y * u->width) + (x + 1);
            int i_bottomleft = ((y + 1) * u->width) + x;
            int i_bottomright = ((y + 1) * u->width) + (x + 1);

            // Pointers to the 4 cells
            unsigned char *topleft = &out->cells[i_topleft];
            unsigned char *topright = &out->cells[i_topright];
            unsigned char *bottomleft = &out->cells[i_bottomleft];
            unsigned char *bottomright = &out->cells[i_bottomright];

            // If all cells are the same, skip processing
            if (*topleft == *topright && *bottomleft == *bottomright && *topleft == *bottomleft)
                continue;

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
            int floor_is_solid = get_cell(u, x, y + 2) >= P_WATER && get_cell(u, x + 1, y + 2) >= P_WATER;

            // Swap of below cells if possible
            // Use different random channel to avoid correlation with top movement
            if (bottom_can_move_horizontally && (floor_is_solid || random_hash(x, y, generation, 5) < WATER_MOVE_HORIZONTAL_CHANCE))
            {
                swap(bottomleft, bottomright);
            }
        }
    }
}