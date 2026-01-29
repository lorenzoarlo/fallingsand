
#include <stdio.h>
#include <stdlib.h>
#include "universe.h"

/**
 * @file graphics.h
 * @brief Functions for exporting the Universe as images.
 */
#ifndef GRAPHICS_H
#define GRAPHICS_H

#define GRAPHICS_FILE_EXTENSION "extension" // to override if needed

/**
 * Simple structure to represent an RGB color.
 */
typedef struct
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
} Color;

/**
 * Mapping function that associates a block type to a color.
 */
Color get_particle_color(unsigned char cell_type);

/**
 * Exports the universe as a PPM (Portable Pixel Map) image.
 *
 * @param u Pointer to the Universe struct.
 * @param filename Output file path (e.g., "output.ppm").
 * @param scale Integer scale factor (e.g., 1 = 1x1 pixel, 4 = 4x4 pixels per block).
 */
void universe_export_image(Universe *u, const char *filename, int scale);

#endif // GRAPHICS_H