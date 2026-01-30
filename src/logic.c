/**
 * @file logic.c
 * @brief Naive implementation of the simulation logic for Falling Sand
 */
#include "simulation.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h> // For perror

/**
 * @brief Handles the behavior of a SAND particle.
 * @param u Pointer to the Universe (current state being modified).
 * @param x Current x-coordinate of the particle.
 * @param y Current y-coordinate of the particle.
 * @param generation Current generation index.
 */
void update_sand(Universe *u, Universe *out, int x, int y, int generation)
{
    int below_y = y + 1;
    unsigned char cell_below = universe_get(u, x, below_y);

    // Check below
    switch (cell_below)
    {
    case P_EMPTY:
    case P_WATER:
    {
        universe_set(out, x, below_y, P_SAND); // Swap below cell with sand
        universe_set(out, x, y, cell_below);   // Insert in cell below
        return;
    }
    case P_SAND:
    case P_WALL:
    {
        int diag_x = (generation % 2 == 0) ? (x - 1) : (x + 1); // Even: Left, Odd: Right
        // Check if the specific diagonal is EMPTY (Strict rule: do not swap with water on diagonal)
        if (universe_get(u, diag_x, below_y) == P_EMPTY)
        {
            universe_set(out, diag_x, below_y, P_SAND); // Move sand to diagonal
            universe_set(out, x, y, P_EMPTY);           // Leave current cell empty
        }
    }
    default:
        return;
    }
}

/**
 * @brief Handles the behavior of a WATER particle.
 * @param u Pointer to the Universe (current state being modified).
 * @param x Current x-coordinate of the particle.
 * @param y Current y-coordinate of the particle.
 * @param generation Current generation index.
 */
void update_water(Universe *u, Universe *out, int x, int y, int generation)
{
    int below_y = y + 1;
    unsigned char cell_below = universe_get(u, x, below_y);

    switch (cell_below)
    {
    case P_EMPTY:
    {
        universe_set(out, x, below_y, P_WATER); // Swap below cell with water
        universe_set(out, x, y, P_EMPTY);       // Insert in cell below
        return;
    }
    case P_WALL:
    case P_SAND:
    {
        int left_x = x - 1;
        int right_x = x + 1;
        int first_x, second_x;
        // Determine diagonal check order based on generation parity
        // In generation even: left first, odd: right first
        first_x = (generation % 2 == 0) ? left_x : right_x;
        second_x = (generation % 2 == 0) ? right_x : left_x;

        // Check first diagonal
        if (universe_get(u, first_x, below_y) == P_EMPTY)
        {
            universe_set(out, first_x, below_y, P_WATER); // Move water to diagonal
            universe_set(out, x, y, P_EMPTY);             // Leave current cell empty
            return;
        }
        // Check second diagonal
        if (universe_get(u, second_x, below_y) == P_EMPTY)
        {

            universe_set(out, second_x, below_y, P_WATER); // Move water to diagonal
            universe_set(out, x, y, P_EMPTY);              // Leave current cell empty
            return;
        }
    }
    default:
        break;
    }

    // If we are here, it means that water couldn't go down or diagonally
    // Try to move horizontally

    int left_x = x - 1;
    int right_x = x + 1;
    // Determine horizontal check order based on generation parity
    int h_first = (generation % 2 == 0) ? right_x : left_x;
    int h_second = (generation % 2 == 0) ? left_x : right_x;

    if (universe_get(u, h_first, y) == P_EMPTY)
    {
        universe_set(out, x, y, P_EMPTY);
        universe_set(out, h_first, y, P_WATER);
    }
    else if (universe_get(u, h_second, y) == P_EMPTY)
    {
        universe_set(out, x, y, P_EMPTY);
        universe_set(out, h_second, y, P_WATER);
    }
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

    // Deep copy the current state to the new universe
    memcpy(new_u->cells, u->cells, u->width * u->height * sizeof(unsigned char));

    int is_odd = generation % 2 == 0;
    int step_x = 1 - (is_odd * 2);
    int start_x = is_odd * (new_u->width - 1);

    // Iteration Order: Top to Bottom
    for (int y = 0; y < new_u->height; y++)
    {
        for (int x = 0; x < new_u->width; x++)
        {
            int real_x = start_x + (x * step_x);
            unsigned char cell = universe_get(u, real_x, y);
            switch (cell)
            {
            case P_SAND:
            {
                update_sand(u, new_u, real_x, y, generation);
                break;
            }
            case P_WATER:
            {
                update_water(u, new_u, real_x, y, generation);
                break;
            }
            default:
                break;
            }
        }
    }

    return new_u;
}