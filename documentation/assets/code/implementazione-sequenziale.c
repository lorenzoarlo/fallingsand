/**
 *  Logic to process a 2x2 block using scalar code
 *  @param x The x coordinate of the top-left cell of the block
 *  @param y The y coordinate of the top-left cell of the block
 *  @param u The universe containing the cells
 *  @param cells The array of cells to process
 *  
 */
static inline void blocklogic(int x, int y, Universe *u, unsigned char *cells, int generation)
{
    // calculate index of the 4 cells
    int i_topleft = (y * u->width) + x;
    int i_topright = (y * u->width) + (x + 1);
    int i_bottomleft = ((y + 1) * u->width) + x;
    int i_bottomright = ((y + 1) * u->width) + (x + 1);

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

    int top_moved = 0;

    // Manage horizontal SAND movement (to make it more noisy when falling)
    int topsand_can_move = *topleft == P_SAND && *topright < P_SAND ||
                           *topright == P_SAND && *topleft < P_SAND;

    int sand_is_falling = *bottomleft < P_SAND && *bottomright < P_SAND;
    if (topsand_can_move && sand_is_falling && random_hash(x, y, generation, 0) < SAND_NOISE_CHANCE)
    {
        swap(topleft, topright);
        top_moved = 1;
    }

    // Manage SAND falling

    // topleft sand particle
    if (!top_moved && *topleft == P_SAND)
    {
        // Can fall down?
        if (*bottomleft < P_SAND)
        {
            if (random_hash(x, y, generation, 1) < SAND_FALL_DOWN_CHANCE)
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
    if (!top_moved && *topright == P_SAND)
    {
        // Can fall down
        if (*bottomright < P_SAND)
        {
            if (random_hash(x, y, generation, 1) < SAND_FALL_DOWN_CHANCE)
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
    int water_dropped = 0;

    // topleft water particle
    if (*topleft == P_WATER)
    {
        // Is the cell below lower density?
        if (*bottomleft < *topleft && random_hash(x, y, generation, 2) < WATER_FALL_DENSITY_CHANCE)
        {
            swap(topleft, bottomleft);
            water_dropped = 1;
        }
        // Can move diagonally?
        else if (*topright < *topleft && *bottomright < *topleft && random_hash(x, y, generation, 3) < WATER_MOVE_DIAGONAL_CHANCE)
        {
            swap(topleft, bottomright);
            water_dropped = 1;
        }
    }

    // topright water particle
    if (*topright == P_WATER)
    {
        // Is the cell below lower density?
        if (*bottomright < *topright && random_hash(x, y, generation, 2) < WATER_FALL_DENSITY_CHANCE)
        {
            swap(topright, bottomright);
            water_dropped = 1;
        } // Can move diagonally?
        else if (*topleft < *topright && *bottomleft < *topright && random_hash(x, y, generation, 3) < WATER_MOVE_DIAGONAL_CHANCE)
        {
            swap(topright, bottomleft);
            water_dropped = 1;
        }
    }

    // Horizontal water movement if not dropped
    if (!water_dropped)
    {
        int top_can_move_horizontally = (*topleft == P_WATER && *topright < P_WATER ||
                                         *topleft < P_WATER && *topright == P_WATER);
        int below_solid = *bottomleft >= P_WATER && *bottomright >= P_WATER;

        if (top_can_move_horizontally && (below_solid || random_hash(x, y, generation, 4) < WATER_MOVE_HORIZONTAL_CHANCE))
        {
            swap(topleft, topright);
        }
    }

    int bottomwater_can_move_horizontally = (*bottomleft == P_WATER && *bottomright < P_WATER ||
                                             *bottomleft < P_WATER && *bottomright == P_WATER);
    // Look if there is solid floor below
    int floor_is_solid = get_cell(u, x, y + 2) >= P_WATER && get_cell(u, x + 1, y + 2) >= P_WATER;

    // Swap of below cells if possible
    // Use different random channel to avoid correlation with top movement
    if (bottomwater_can_move_horizontally && (floor_is_solid || random_hash(x, y, generation, 5) < WATER_MOVE_HORIZONTAL_CHANCE))
    {
        swap(bottomleft, bottomright);
    }
}