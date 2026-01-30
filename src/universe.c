/**
 * @file universe.c
 * @brief Implements functions for managing 2D grid of falling sand simulation.
 */
#include "universe.h"

#include <stdlib.h> // For malloc, calloc, free
#include <stdio.h>  // For perror

Universe *universe_create(int width, int height)
{
    // Validate dimensions
    if (width <= 0 || height <= 0)
    {
        perror("universe_create -> Error, invalid universe dimensions");
        return NULL;
    }

    Universe *u = (Universe *)malloc(sizeof(Universe));

    // Failed to allocate memory for Universe struct
    if (!u)
    {
        perror("universe_create -> Error allocating memory for Universe struct");
        return NULL;
    }

    u->width = width;
    u->height = height;

    // Using calloc to initialize all cells to P_EMPTY (0)
    u->cells = (unsigned char *)calloc(width * height, sizeof(unsigned char));

    // Failed to allocate memory for cells
    if (!u->cells)
    {
        perror("universe_create -> Error allocating memory for Universe cells");
        free(u);
        return NULL;
    }

    return u;
}

void universe_destroy(Universe *u)
{
    if (u)
    {
        if (u->cells)
        {
            free(u->cells);
        }
        free(u);
    }
}

int universe_out_of_bounds(Universe *universe, int x, int y)
{
    return (x < 0 || x >= universe->width || y < 0 || y >= universe->height);
}

unsigned char universe_get(Universe *u, int x, int y)
{
    // If outside bounds, return P_WALL
    if (universe_out_of_bounds(u, x, y))
    {
        /**
         * Out of bounds reads return P_WALL
         */
        return P_WALL;
    }

    return u->cells[UINDEX(x, y, u->width)];
}

void universe_set(Universe *u, int x, int y, unsigned char value)
{
    // If outside bounds, do nothing
    if (universe_out_of_bounds(u, x, y))
    {
        /**
         * Out of bounds writes are ignored
         */
        return;
    }

    u->cells[UINDEX(x, y, u->width)] = value;
}

