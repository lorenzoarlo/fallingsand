/**
 * @file io.c
 * @brief Implementation of file I/O functions for the falling sand simulation.
 * It defines a new file format ".sand" to store simulation frames.
 */
#include "io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


FILE *io_create_file(const char *filename, int width, int height)
{
    // Create and open the file for writing in binary mode
    FILE *file = fopen(filename, "wb");
    if (!file)
    {
        perror("io_create_file -> Error creating file!");
        return NULL;
    }

    // Write magic number (to identify file type)
    fwrite(FILE_MAGIC, sizeof(char), 4, file);

    // Write width and height as 4-byte integers
    fwrite(&width, sizeof(int), 1, file);
    fwrite(&height, sizeof(int), 1, file);
    int frames = 0;
    fwrite(&frames, sizeof(int), 1, file); // Initial number of frames is 0

    return file;
}

void io_append_frame(FILE *file, Universe *universe)
{
    // Validate inputs
    if (!file || !universe)
    {
        perror("io_append_frame -> Invalid file or universe!");
        return;
    }

    fwrite(universe->cells, sizeof(unsigned char), universe->width * universe->height, file);

    // Move to the frame count position
    fseek(file, FRAME_POSITION_OFFSET, SEEK_SET);

    int frames = 0;
    // Read the current frame count
    fread(&frames, sizeof(int), 1, file);
    frames += 1;

    // Move back to the frame count position
    fseek(file, FRAME_POSITION_OFFSET, SEEK_SET);

    // Write the updated frame count
    fwrite(&frames, sizeof(int), 1, file);

    // Move back to the end of the file
    fseek(file, 0, SEEK_END);
}

FILE *io_open_read(const char *filename, int *width, int *height, int *num_frames)
{
    // Open the file for reading in binary mode
    FILE *file = fopen(filename, "rb");
    if (!file)
    {
        perror("io_open_read -> Error opening file!");
        return NULL;
    }

    // Read and validate magic number
    char magic[5];
    fread(magic, sizeof(char), 4, file);
    if (strncmp(magic, FILE_MAGIC, 4) != 0)
    {
        perror("io_open_read -> Invalid file format (wrong magic number)!");
        fclose(file);
        return NULL;
    }

    // Read width, height, and number of frames
    fread(width, sizeof(int), 1, file);
    fread(height, sizeof(int), 1, file);
    fread(num_frames, sizeof(int), 1, file);

    return file;
}

int io_read_frame(FILE *file, Universe *u)
{
    // Validate inputs
    if (!file || !u)
    {
        perror("io_read_frame -> Invalid file or universe!");
        return -1;
    }

    size_t read = fread(u->cells, sizeof(unsigned char), u->width * u->height, file);
    if (read != (size_t)(u->width * u->height))
    {
        perror("io_read_frame -> Error reading frame data!");
        return -1;
    }

    return 0;
}