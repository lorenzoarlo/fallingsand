/**
 * @file io.c
 * @brief Implementation of file I/O functions for the falling sand simulation.
 * It defines a new file format ".sand" to store simulation frames using bit-packing compression.
 */
#include "io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Helper macro to calculate packed size
// We need 2 bits per pixel, so 4 pixels fit in 1 byte.
#define PACKED_SIZE(w, h) (((w) * (h) + 3) / 4)

FILE *io_create_file(const char *filename, int width, int height)
{
    // Create and open the file for writing in binary mode
    FILE *file = fopen(filename, "wb");
    if (!file)
    {
        sprintf(stderr, "io_create_file -> Error creating file \"%s\"!\n", filename);
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

    int total_pixels = universe->width * universe->height;
    int packed_size = PACKED_SIZE(universe->width, universe->height);

    // Allocate buffer for packed data (initialized to 0)
    unsigned char *packed_data = calloc(packed_size, sizeof(unsigned char));
    if (!packed_data)
    {
        perror("io_append_frame -> Allocation failed!");
        return;
    }

    // Pack 4 pixels into 1 byte
    // Pixel values (assumed 0-3) take 2 bits each
    for (int i = 0; i < total_pixels; i++)
    {
        int byte_index = i / 4;
        int bit_shift = (i % 4) * 2; // Shift by 0, 2, 4, or 6 bits
        
        // Ensure value is within 2 bits and shift it to position
        packed_data[byte_index] |= ((universe->cells[i] & 0x03) << bit_shift);
    }

    // Write the compressed buffer
    fwrite(packed_data, sizeof(unsigned char), packed_size, file);
    
    // Clean up
    free(packed_data);

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
        sprintf(stderr,"io_open_read -> Error opening file \"%s\"!\n", filename);
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

    int total_pixels = u->width * u->height;
    int packed_size = PACKED_SIZE(u->width, u->height);

    // Allocate temp buffer for reading packed data
    unsigned char *packed_data = malloc(packed_size);
    if (!packed_data)
    {
        perror("io_read_frame -> Allocation failed!");
        return -1;
    }

    size_t read = fread(packed_data, sizeof(unsigned char), packed_size, file);
    if (read != (size_t)packed_size)
    {
        perror("io_read_frame -> Error reading frame data!");
        free(packed_data);
        return -1;
    }

    // Unpack data: Extract 4 pixels from each byte
    for (int i = 0; i < total_pixels; i++)
    {
        int byte_index = i / 4;
        int bit_shift = (i % 4) * 2;
        
        // Extract 2 bits and assign to cell
        u->cells[i] = (packed_data[byte_index] >> bit_shift) & 0x03;
    }

    free(packed_data);
    return 0;
}