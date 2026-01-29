/**
 * @file universe.c
 * @brief Implements functions for managing 2D grid of falling sand simulation.
 */
#include "universe.h"
#include <stdlib.h>

Universe *universe_create(int width, int height)
{
    // Validate dimensions
    if (width <= 0 || height <= 0)
    {
        return NULL;
    }

    Universe *u = (Universe *)malloc(sizeof(Universe));

    // Failed to allocate memory for Universe struct
    if (!u)
    {
        return NULL;
    }

    u->width = width;
    u->height = height;

    // Using calloc to initialize all cells to P_EMPTY (0)
    u->cells = (unsigned char *)calloc(width * height, sizeof(unsigned char));

    // Failed to allocate memory for cells
    if (!u->cells)
    {
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

unsigned char universe_get(Universe *u, int x, int y)
{
    // If outside bounds, return P_WALL
    if (x < 0 || x >= u->width || y < 0 || y >= u->height)
    {
        return P_WALL;
    }

    return u->cells[UINDEX(x, y, u->width)];
}

void universe_set(Universe *u, int x, int y, unsigned char value)
{
    // If outside bounds, do nothing
    if (x < 0 || x >= u->width || y < 0 || y >= u->height)
    {
        return;
    }

    u->cells[UINDEX(x, y, u->width)] = value;
}

void universe_swap(Universe *u, int x1, int y1, int x2, int y2)
{
    // Validate coordinates
    int valid1 = (x1 >= 0 && x1 < u->width && y1 >= 0 && y1 < u->height);
    int valid2 = (x2 >= 0 && x2 < u->width && y2 >= 0 && y2 < u->height);

    if (valid1 && valid2)
    {
        // Swap the values at the two coordinates
        unsigned char temp = u->cells[UINDEX(x1, y1, u->width)];
        u->cells[UINDEX(x1, y1, u->width)] = u->cells[UINDEX(x2, y2, u->width)];
        u->cells[UINDEX(x2, y2, u->width)] = temp;
    }
}