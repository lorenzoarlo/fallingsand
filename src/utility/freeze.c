/**
 * @file freeze.c
 * Simple logic to verify if outside algorithms works.
 * The logic is very simple: just copy the current state to the next state.
 */
#include "../simulation.h"

#include <stdlib.h>

/**
 * Calculates the next state of the universe based on the current state.
 * This is a dump implementation that just copies the current state.
 */
Universe *next(Universe *u, int generation)
{
    // Create a new universe initialized as empty (or copy of u)
    Universe *new_u = universe_create(u->width, u->height);

    // Simple logic: Copy everything as is (No physics)
    for (int y = 0; y < u->height; y++)
    {
        for (int x = 0; x < u->width; x++)
        {
            universe_set(new_u, x, y, universe_get(u, x, y));
        }
    }
    return new_u;
}