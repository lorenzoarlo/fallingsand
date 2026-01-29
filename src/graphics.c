/**
 * @file graphics.h
 * @brief Functions for exporting the Universe as images.
 */
#include "graphics.h"
#include <stdlib.h>
#include <string.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "utility/stb_image_write.h" // Include the stb_image_write header

#define GRAPHICS_FILE_EXTENSION "png"

/**
 * Mapping function that associates a block type to a color.
 */
Color get_particle_color(unsigned char cell_type)
{
    Color c;
    switch (cell_type)
    {
    case P_SAND:
        c.r = 237;
        c.g = 201;
        c.b = 175;
        break; // Sand yellow
    case P_WATER:
        c.r = 0;
        c.g = 100;
        c.b = 255;
        break; // Water blue
    case P_WALL:
        c.r = 100;
        c.g = 100;
        c.b = 100;
        break; // Dark gray
    case P_EMPTY:
        c.r = 255;
        c.g = 255;
        c.b = 255;
        break; // White
    default:
        c.r = 255;
        c.g = 0;
        c.b = 255;
        break; // Magenta (Error)
    }
    return c;
}

/**
 * Exports the universe as a PNG image using stb_image_write.
 *
 * @param u Pointer to the Universe struct.
 * @param filename Output file path (e.g., "output.png").
 * @param scale Integer scale factor (e.g., 1 = 1x1 pixel, 4 = 4x4 pixels per block).
 */
void universe_export_image(Universe *u, const char *filename, int scale)
{
    // Check parameters
    if (u == NULL || filename == NULL || scale < 1)
    {
        perror("Error, invalid parameters for image export.\n");
        return;
    }
    // Verify extension
    size_t len_str = strlen(filename);
    size_t len_suffix = strlen(GRAPHICS_FILE_EXTENSION);

    if (len_str < len_suffix || (strcmp(filename + len_str - len_suffix, GRAPHICS_FILE_EXTENSION) != 0))
    {
        fprintf(stderr, "Error, invalid filename extension for image export. It should end with %s\n", GRAPHICS_FILE_EXTENSION);
        return;
    }

    // Calculate
    int img_width = u->width * scale;
    int img_height = u->height * scale;
    int channels = 3; // RGB

    int buffer_size = img_width * img_height * channels;
    unsigned char *image_data = (unsigned char *)malloc(buffer_size);

    if (image_data == NULL)
    {
        perror("Error allocating memory for image export");
        return;
    }

    // filling image data
    for (int y = 0; y < u->height; y++)
    {
        for (int x = 0; x < u->width; x++)
        {
            // Get cell type and corresponding color
            unsigned char cell = universe_get(u, x, y);
            Color color = get_particle_color(cell);

            // Fill the scaled pixels
            for (int dy = 0; dy < scale; dy++)
            {
                for (int dx = 0; dx < scale; dx++)
                {
                    int pixel_x = (x * scale) + dx;
                    int pixel_y = (y * scale) + dy;

                    int index = (pixel_y * img_width + pixel_x) * channels;

                    image_data[index + 0] = (unsigned char)color.r;
                    image_data[index + 1] = (unsigned char)color.g;
                    image_data[index + 2] = (unsigned char)color.b;
                }
            }
        }
    }

    // Write PNG using stb_image_write
    int stride_in_bytes = img_width * channels;

    int result = stbi_write_png(filename, img_width, img_height, channels, image_data, stride_in_bytes);

    if (result == 0)
    {
        fprintf(stderr, "Error: Failed to write PNG file to %s\n", filename);
    }
    else
    {
        printf("Image saved in %s with scale %d (Format: PNG)\n", filename, scale);
    }

    free(image_data);
}