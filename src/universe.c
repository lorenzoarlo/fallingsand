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






