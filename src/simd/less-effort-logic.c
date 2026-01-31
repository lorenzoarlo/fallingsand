/**
 * @file less-effort-logic.c
 * @brief More studied implementation of the simulation logic for Falling Sand
 * It will use some optimizations (no simd yet) to reduce unnecessary computations.
 * For example:
 * - it avoids useless checks of out of bound (everywhere except when is needed to understand if it is a wall)
 * - it will skip the empty cells update (by default they are empty)
 */
#include "../simulation.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h> // For perror

/**
 * @brief Wrapper structure to hold the current universe and a clock buffer.
 */
typedef struct WrapUniverse
{
    Universe *current;
    unsigned char *clock;
} WrapUniverse;

/**
 * @brief Sets the particle type at the specified coordinates in the Universe.
 * It does NOT check for out of bounds for performance reasons.
 * @param universe Pointer to the Universe.
 * @param x The x-coordinate.
 * @param y The y-coordinate.
 * @param value The particle type to set.
 * NOTES: This is a less effort version that does not check for out of bounds.
 */
static inline void less_effort_universe_set(Universe *u, int x, int y, unsigned char value)
{
    u->cells[UINDEX(x, y, u->width)] = value;
}

/**
 * @brief Checks if the cell at (x, y) has already been updated in the current generation.
 * @param in Pointer to the WrapUniverse containing the clock buffer.
 * @param x The x-coordinate of the cell.
 * @param y The y-coordinate of the cell.
 *
 * NOTES: Out of bounds check is avoided here for performance.
 */
static inline int already_updated(WrapUniverse *in, int x, int y)
{
    return in->clock[UINDEX(x, y, in->current->width)] == 1;
}

/**
 * @brief Marks the cell at (x, y) as updated in the current generation.
 * @param in Pointer to the WrapUniverse containing the clock buffer.
 * @param x The x-coordinate of the cell.
 * @param y The y-coordinate of the cell.
 *
 * NOTES: Out of bounds check is avoided here for performance.
 */
static inline void update_cellsclock(WrapUniverse *in, int x, int y)
{
    in->clock[UINDEX(x, y, in->current->width)] = 1;
}

/**
 * @brief Gets the particle type at (x, y) by verifying if already updated.
 * @param in Pointer to the WrapUniverse containing the current universe.
 * @param out Pointer to the Universe representing the next state.
 * @param x The x-coordinate.
 * @param y The y-coordinate.
 * @return The particle type at the specified coordinates.
 */
static inline unsigned char universe_wrap_get(WrapUniverse *in, Universe *out, int x, int y)
{
    return already_updated(in, x, y) ? universe_get(out, x, y) : universe_get(in->current, x, y);
}

/**
 * @brief Handles the behavior of a SAND particle.
 * @param u Pointer to the Universe (current state being modified).
 * @param x Current x-coordinate of the particle.
 * @param y Current y-coordinate of the particle.
 * @param generation Current generation index.
 */
static inline void update_sand(WrapUniverse *in, Universe *out, int x, int y, int generation)
{
    int below_y = y + 1;
    unsigned char cell_below = universe_wrap_get(in, out, x, below_y);

    // Fall through EMPTY space
    if (cell_below == P_EMPTY)
    {
        less_effort_universe_set(out, x, below_y, P_SAND);
        less_effort_universe_set(out, x, y, P_EMPTY);
        update_cellsclock(in, x, below_y);
        return;
    }

    // Verify diagonals when blocked by SAND or WALL
    int dir = (generation % 2 == 0) ? 1 : -1;
    int diag1_x = x - dir;
    int diag2_x = x + dir;

    // This means that there is something else below (SAND or WALL or WATER)
    // Check first diagonal
    if (universe_wrap_get(in, out, diag1_x, below_y) == P_EMPTY)
    {
        less_effort_universe_set(out, diag1_x, below_y, P_SAND);
        less_effort_universe_set(out, x, y, P_EMPTY);
        update_cellsclock(in, diag1_x, below_y);
        return;
    }
    // Check second diagonal
    if (universe_wrap_get(in, out, diag2_x, below_y) == P_EMPTY)
    {
        less_effort_universe_set(out, diag2_x, below_y, P_SAND);
        less_effort_universe_set(out, x, y, P_EMPTY);
        update_cellsclock(in, diag2_x, below_y);
        return;
    }

    // Only if the cell below is WATER, we have to manage the swap (density)
    if (cell_below == P_WATER)
    {
        // To simulate viscosity, we make SAND stay still in half of the cases
        // Unless we will have a bad tendency to have WATER going up too fast
        if ((x + y + generation) % 2 != 0)
        {
            less_effort_universe_set(out, x, y, P_SAND); // Stay in place
            return;
        }

        // Else swap positions
        less_effort_universe_set(out, x, below_y, P_SAND);
        less_effort_universe_set(out, x, y, P_WATER); // L'acqua sale
        update_cellsclock(in, x, below_y);
        return;
    }

    // If we are here, it means that SAND couldn't go down or diagonally
    less_effort_universe_set(out, x, y, P_SAND);
}
/**
 * @brief Handles the behavior of a WATER particle.
 * @param u Pointer to the Universe (current state being modified).
 * @param x Current x-coordinate of the particle.
 * @param y Current y-coordinate of the particle.
 * @param generation Current generation index.
 */
void update_water(WrapUniverse *in, Universe *out, int x, int y, int generation)
{
    // Unwrap
    Universe *u = in->current;

    int below_y = y + 1;
    unsigned char cell_below = universe_wrap_get(in, out, x, below_y);

    if (cell_below == P_EMPTY)
    {
        less_effort_universe_set(out, x, below_y, P_WATER); // Swap below cell with water
        less_effort_universe_set(out, x, y, P_EMPTY);       // Insert in cell below
        update_cellsclock(in, x, below_y);                  // Mark as updated also the below cell
        return;
    }

    if (cell_below == P_WALL || cell_below == P_WATER || cell_below == P_SAND)
    {
        int left_x = x - 1;
        int right_x = x + 1;
        int first_x, second_x;
        // Determine diagonal check order based on generation parity
        // In generation even: left first, odd: right first
        first_x = (generation % 2 == 0) ? left_x : right_x;
        second_x = (generation % 2 == 0) ? right_x : left_x;

        // Check first diagonal
        if (universe_wrap_get(in, out, first_x, below_y) == P_EMPTY)
        {
            less_effort_universe_set(out, first_x, below_y, P_WATER); // Move water to diagonal
            less_effort_universe_set(out, x, y, P_EMPTY);             // Leave current cell empty
            update_cellsclock(in, first_x, below_y);                  // Mark as updated also the other cell
            return;
        }
        // Check second diagonal
        if (universe_wrap_get(in, out, second_x, below_y) == P_EMPTY)
        {
            less_effort_universe_set(out, second_x, below_y, P_WATER); // Move water to diagonal
            less_effort_universe_set(out, x, y, P_EMPTY);              // Leave current cell empty
            update_cellsclock(in, second_x, below_y);                  // Mark as updated also the other cell
            return;
        }
    }

    // If we are here, it means that water couldn't go down or diagonally
    // Try to move horizontally

    int left_x = x - 1;
    int right_x = x + 1;
    // Determine horizontal check order based on generation parity
    int h_first = (generation % 2 == 0) ? right_x : left_x;
    int h_second = (generation % 2 == 0) ? left_x : right_x;

    if (universe_wrap_get(in, out, h_first, y) == P_EMPTY)
    {
        less_effort_universe_set(out, h_first, y, P_WATER);
        less_effort_universe_set(out, x, y, P_EMPTY);
        update_cellsclock(in, h_first, y); // Mark as updated also the moved cell
        return;
    }

    if (universe_wrap_get(in, out, h_second, y) == P_EMPTY)
    {
        less_effort_universe_set(out, x, y, P_EMPTY);
        less_effort_universe_set(out, h_second, y, P_WATER);
        update_cellsclock(in, h_second, y); // Mark as updated also the moved cell
        return;
    }

    less_effort_universe_set(out, x, y, P_WATER); // Stay in place
}



Universe *next(Universe *u, int generation)
{
    // Create a new universe to hold the next state.
    Universe *new_u = universe_create(u->width, u->height);
    if (!new_u)
    {
        perror("next -> Error creating new universe for next generation");
        return NULL;
    }

    // Copy the universe to initialize (all empty cells and walls will remain the same)
    memcpy(new_u->cells, u->cells, u->width * u->height * sizeof(unsigned char));

    int is_odd = generation % 2 != 0;
    int step_x = 1 - (is_odd * 2);
    int start_x = is_odd * (new_u->width - 1);

    // Create clock buffer
    unsigned char *clock_buffer = (unsigned char *)calloc(u->width * u->height, sizeof(unsigned char));
    WrapUniverse wrap = {u, clock_buffer};

    // Iteration Order: Top to Bottom
    for (int y = 0; y < new_u->height; y++)
    {
        for (int x = 0; x < new_u->width; x++)
        {
            int real_x = start_x + (x * step_x);
            // Update only if not already updated
            if (already_updated(&wrap, real_x, y))
            {
                continue; // Skip already updated cells
            }
            // Mark as updated
            update_cellsclock(&wrap, real_x, y);

            unsigned char cell = universe_get(u, real_x, y);
            switch (cell)
            {
            case P_SAND:
            {
                update_sand(&wrap, new_u, real_x, y, generation);
                break;
            }
            case P_WATER:
            {
                update_water(&wrap, new_u, real_x, y, generation);
                break;
            }
            case P_WALL:  // NOTES: Walls are already set in the memcpy
            case P_EMPTY: // NOTES: Empty are already set in the memcpy
            {
                // No action needed, already set
                break;
            }
            default:
                break;
            }
        }
    }

    free(clock_buffer);
    return new_u;
}