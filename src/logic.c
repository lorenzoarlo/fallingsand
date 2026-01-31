/**
 * @file logic.c
 * @brief Naive implementation of the simulation logic for Falling Sand
 */
#include "simulation.h"
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
 * @brief Checks if the cell at (x, y) has already been updated in the current generation.
 * @param in Pointer to the WrapUniverse containing the clock buffer.
 * @param x The x-coordinate of the cell.
 * @param y The y-coordinate of the cell.
 */
static inline int already_updated(WrapUniverse *in, int x, int y)
{
    if (universe_out_of_bounds(in->current, x, y))
    {
        return 0; // Out of bounds
    }
    return in->clock[UINDEX(x, y, in->current->width)] == 1;
}

/**
 * @brief Marks the cell at (x, y) as updated in the current generation.
 * @param in Pointer to the WrapUniverse containing the clock buffer.
 * @param x The x-coordinate of the cell.
 * @param y The y-coordinate of the cell.
 */
static inline void update_cellsclock(WrapUniverse *in, int x, int y)
{
    if (universe_out_of_bounds(in->current, x, y))
    {
        return; // Out of bounds
    }
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
        universe_set(out, x, below_y, P_SAND);
        universe_set(out, x, y, P_EMPTY);
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
        universe_set(out, diag1_x, below_y, P_SAND);
        universe_set(out, x, y, P_EMPTY);
        update_cellsclock(in, diag1_x, below_y);
        return;
    }
    // Check second diagonal
    if (universe_wrap_get(in, out, diag2_x, below_y) == P_EMPTY)
    {
        universe_set(out, diag2_x, below_y, P_SAND);
        universe_set(out, x, y, P_EMPTY);
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
            universe_set(out, x, y, P_SAND); // Stay in place
            return;
        }

        // Else swap positions
        universe_set(out, x, below_y, P_SAND);
        universe_set(out, x, y, P_WATER); // L'acqua sale
        update_cellsclock(in, x, below_y);
        return;
    }

    // If we are here, it means that SAND couldn't go down or diagonally
    universe_set(out, x, y, P_SAND);
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
        universe_set(out, x, below_y, P_WATER); // Swap below cell with water
        universe_set(out, x, y, P_EMPTY);       // Insert in cell below
        update_cellsclock(in, x, below_y);      // Mark as updated also the below cell
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
            universe_set(out, first_x, below_y, P_WATER); // Move water to diagonal
            universe_set(out, x, y, P_EMPTY);             // Leave current cell empty
            update_cellsclock(in, first_x, below_y);      // Mark as updated also the other cell
            return;
        }
        // Check second diagonal
        if (universe_wrap_get(in, out, second_x, below_y) == P_EMPTY)
        {
            universe_set(out, second_x, below_y, P_WATER); // Move water to diagonal
            universe_set(out, x, y, P_EMPTY);              // Leave current cell empty
            update_cellsclock(in, second_x, below_y);      // Mark as updated also the other cell
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
        universe_set(out, h_first, y, P_WATER);
        universe_set(out, x, y, P_EMPTY);
        update_cellsclock(in, h_first, y); // Mark as updated also the moved cell
        return;
    }

    if (universe_wrap_get(in, out, h_second, y) == P_EMPTY)
    {
        universe_set(out, x, y, P_EMPTY);
        universe_set(out, h_second, y, P_WATER);
        update_cellsclock(in, h_second, y); // Mark as updated also the moved cell
        return;
    }

    universe_set(out, x, y, P_WATER); // Stay in place
}

static inline void update_wall(WrapUniverse *in, Universe *out, int x, int y, int generation)
{
    // WALL particles do not move; they remain static.
    universe_set(out, x, y, P_WALL);
}

static inline void update_empty(WrapUniverse *in, Universe *out, int x, int y, int generation)
{
    // EMPTY cells remain empty.
    universe_set(out, x, y, P_EMPTY);
}

void next(Universe *u, Universe* out, int generation)
{
    int is_odd = generation % 2 != 0;
    int step_x = 1 - (is_odd * 2);
    int start_x = is_odd * (u->width - 1);

    // Create clock buffer
    unsigned char *clock_buffer = (unsigned char *)calloc(u->width * u->height, sizeof(unsigned char));
    WrapUniverse wrap = {u, clock_buffer};

    // Iteration Order: Top to Bottom
    for (int y = 0; y < u->height; y++)
    {
        for (int x = 0; x < u->width; x++)
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
                update_sand(&wrap, out, real_x, y, generation);
                break;
            }
            case P_WATER:
            {
                update_water(&wrap, out, real_x, y, generation);
                break;
            }
            case P_WALL:
            {
                update_wall(&wrap, out, real_x, y, generation);
                break;
            }
            case P_EMPTY:
            {
                update_empty(&wrap, out, real_x, y, generation);
                break;
            }
            default:
                break;
            }
        }
    }

    free(clock_buffer);
}