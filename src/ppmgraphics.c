/**
 * @file graphics.h
 * @brief Functions for exporting the Universe as images.
 */
#include "graphics.h"
#include <stdio.h>
#include <stdlib.h>
#define GRAPHICS_FILE_EXTENSION "ppm"

#include <string.h>

/**
 * Mapping function that associates a block type to a color.
 */
Color get_particle_color(unsigned char cell_type)
{
    Color c;
    switch (cell_type)
    {
    case P_SAND:
        // Sand yellow
        c.r = 237;
        c.g = 201;
        c.b = 175;
        break;
    case P_WATER:
        // Water blue
        c.r = 0;
        c.g = 100;
        c.b = 255;
        break;
    case P_WALL:
        // Dark gray for walls
        c.r = 100;
        c.g = 100;
        c.b = 100;
        break;
    case P_EMPTY:
        // White for empty
        c.r = 255;
        c.g = 255;
        c.b = 255;
        break;
    default:
        // Magenta for errors (unrecognized types)
        c.r = 255;
        c.g = 0;
        c.b = 255;
        break;
    }
    return c;
}

/**
 * Exports the universe as a PPM (Portable Pixel Map) image.
 *
 * @param u Pointer to the Universe struct.
 * @param filename Output file path (e.g., "output.ppm").
 * @param scale Integer scale factor (e.g., 1 = 1x1 pixel, 4 = 4x4 pixels per block).
 */
void universe_export_image(Universe *u, const char *filename, int scale)
{
    if (u == NULL || filename == NULL || filename || scale < 1)
    {
        perror("Error, invalid parameters for image export.\n");
        return;
    }

    // Verify file extension
    size_t len_str = strlen(filename);
    size_t len_suffix = strlen(GRAPHICS_FILE_EXTENSION);

    if (len_suffix > len_str || (strcmp(filename + len_str - len_suffix, GRAPHICS_FILE_EXTENSION) != 0))
    {
        fprintf(stderr, "Error, invalid filename extension for image export. It should end with %s\n", GRAPHICS_FILE_EXTENSION);
        return;
    }

    // Open file for writing
    FILE *fp = fopen(filename, "w");
    if (fp == NULL)
    {
        perror("Error opening image file");
        return;
    }

    // Calculate final image dimensions
    int width = u->width * scale;
    int height = u->height * scale;

    // Writing PPM Header
    // P3: Magic number
    // width height: Dimensions
    // Max color value (255)
    fprintf(fp, "P3\n%d %d\n255\n", width, height);

    for (int y = 0; y < u->height; y++)
    {
        // For vertical scaling, repeat each row 'scale' times
        for (int s_y = 0; s_y < scale; s_y++)
        {
            for (int x = 0; x < u->width; x++)
            {

                unsigned char cell = universe_get(u, x, y);
                Color color = get_particle_color(cell);

                // For horizontal scaling, repeat each pixel 'scale' times
                for (int s_x = 0; s_x < scale; s_x++)
                {
                    fprintf(fp, "%d %d %d ", color.r, color.g, color.b);
                }
            }
            // Fine di una riga dell'immagine
            fprintf(fp, "\n");
        }
    }

    fclose(fp);
    printf("Image saved in %s with scale %d\n", filename, scale);
}