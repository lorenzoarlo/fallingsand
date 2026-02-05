/**
 * @file test.c
 * @brief Implementation of test functionalities, including comparison of .sand files.
 */
#include "test.h"

#include "io.h"     // For io_open_read, io_read_frame
#include <stdio.h>  // For fprintf, stderr
#include <stdlib.h> // For malloc, free
int test_sand(const char *filename_a, const char *filename_b, Universe ***out_diff_sequence, int *out_frames)
{
    if (out_frames == NULL)
    {
        fprintf(stderr, "test_sand -> Output frames variable not valid");
        return PARAMETERS_NO_VALID_ERROR;
    }
    int w_a, h_a, f_a; // width, height, frames for file A
    int w_b, h_b, f_b; // width, height, frames for file B

    // Open files
    FILE *fa = io_open_read(filename_a, &w_a, &h_a, &f_a);
    if (!fa)
    {
        fprintf(stderr, "test_sand -> Error reading file %s", filename_a);
        return FILE_NULL_ERROR;
    }

    FILE *fb = io_open_read(filename_b, &w_b, &h_b, &f_b);
    if (!fb)
    {
        fprintf(stderr, "test_sand -> Error reading file %s", filename_b);
        // Closing previously opened file
        fclose(fa);
        return FILE_NULL_ERROR;
    }

    if (w_a != w_b || h_a != h_b || f_a != f_b)
    {
        fprintf(stderr, "test_sand -> Number of width, height or frames do not match between the two files");
        fprintf(stderr, "test_sand -> File A - Width: %d, Height: %d, Frames: %d", w_a, h_a, f_a);
        fprintf(stderr, "test_sand -> File B - Width: %d, Height: %d, Frames: %d", w_b, h_b, f_b);

        fclose(fa);
        fclose(fb);
        return PARAMETERS_MISMATCH_ERROR;
    }

    *out_frames = f_a;

    // Allocate memory for the differences sequence;
    Universe **diffs = (Universe **)calloc(f_a, sizeof(Universe *));
    if (!diffs)
    {
        fclose(fa);
        fclose(fb);
        return MEMORY_FAIL_ERROR;
    }

    // Temporary variables
    Universe *temp_a = universe_create(w_a, h_a);
    Universe *temp_b = universe_create(w_b, h_b);

    if (!temp_a || !temp_b)
    {
        if (temp_a)
        {
            universe_destroy(temp_a);
        }
        if (temp_b)
        {
            universe_destroy(temp_b);
        }
        // Clean resources
        free(diffs);
        fclose(fa);
        fclose(fb);
        perror("test_sand -> Memory allocation failed for temporary universes");
        return MEMORY_FAIL_ERROR;
    }

    int success = 1;
    // iterate through frames
    for (int i = 0; i < f_a; i++)
    {
        diffs[i] = universe_create(w_a, h_a);
        if (!diffs[i])
        {
            perror("test_sand -> Memory allocation failed for difference universe");
            success = 0;
            break;
        }

        // Reading frames
        if (io_read_frame(fa, temp_a) != 0 || io_read_frame(fb, temp_b) != 0)
        {
            fprintf(stderr, "test_sand -> Error reading frames from test files");
            success = 0;
            break;
        }

        // Pixel equals check
        for (int y = 0; y < h_a; y++)
        {
            for (int x = 0; x < w_a; x++)
            {
                unsigned char a = universe_get(temp_a, x, y);
                unsigned char b = universe_get(temp_b, x, y);
                // Mark difference or empty
                universe_set(diffs[i], x, y, (a == b) ? P_EMPTY : P_ERROR);
                if (a != b)
                {
                    success = 0;
                }
            }
        }
    }

    // Cleaning
    universe_destroy(temp_a);
    universe_destroy(temp_b);
    fclose(fa);
    fclose(fb);

    if (success)
    {
        // File identici: libera tutto e ritorna successo
        for (int k = 0; k < *out_frames; k++)
        {
            if (diffs[k])
            {
                universe_destroy(diffs[k]);
            }
        }
        free(diffs);
        *out_diff_sequence = NULL;
        *out_frames = 0;
        return 0;
    }
    else
    {
        *out_frames = f_a;
        *out_diff_sequence = diffs;
        return 1;
    }
}