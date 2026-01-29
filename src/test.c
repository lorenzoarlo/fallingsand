/**
 * @file test.c
 * @brief Implementation of test functionalities, including comparison of .sand files.
 */
#include "test.h"

int test_sand(const char *filename_a, const char *filename_b, Universe ***out_diff_sequence, int *out_frames)
{
    if (out_frames == NULL)
    {
        perror("Output frames variable not valid");
        return PARAMETERS_NO_VALID_ERROR;
    }
    int w_a, h_a, f_a; // width, height, frames for file A
    int w_b, h_b, f_b; // width, height, frames for file B

    // 1. Apertura dei file
    FILE *fa = io_open_read(filename_a, &w_a, &h_a, &f_a);
    if (fa == NULL)
    {
        fprintf(stderr,"Error reading file %s", filename_a);
        return FILE_NULL_ERROR;
    }

    FILE *fb = io_open_read(filename_b, &w_b, &h_b, &f_b);
    if (fb == NULL)
    {
        fprintf(stderr,"Error reading file %s", filename_b);
        // Closing previously opened file
        fclose(fa);
        return FILE_NULL_ERROR;
    }

    if (w_a != w_b || h_a != h_b || f_a != f_b)
    {
        fprintf(stderr, "Number of width, height or frames do not match between the two files");
        fprintf(stderr, "File A - Width: %d, Height: %d, Frames: %d", w_a, h_a, f_a);
        fprintf(stderr, "File B - Width: %d, Height: %d, Frames: %d", w_b, h_b, f_b);

        fclose(fa);
        fclose(fb);
        return PARAMETERS_MISMATCH_ERROR;
    }

    // Allocate memory for the differences sequence
    Universe **diffs = (Universe **)malloc(sizeof(Universe *) * f_a);
    if (diffs == NULL)
    {
        fclose(fa);
        fclose(fb);
        return MEMORY_FAIL_ERROR;
    }

    // Temporary variables
    Universe *temp_a = universe_create(w_a, h_a);
    Universe *temp_b = universe_create(w_b, h_b);

    if (temp_a == NULL || temp_b == NULL)
    {

        if (temp_a)
        {
            universe_destroy(temp_a);
        }
        if (temp_b)
        {
            universe_destroy(temp_b);
        }
        free(diffs);
        fclose(fa);
        fclose(fb);
        return MEMORY_FAIL_ERROR;
    }

    int success = 1;
    for (int i = 0; i < f_a; i++)
    {

        diffs[i] = universe_create(w_a, h_a);

        // Leggiamo i frame dai file
        if (io_read_frame(fa, temp_a) != 0 || io_read_frame(fb, temp_b) != 0)
        {
            perror("Error reading frames from files");
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

                universe_set(diffs[i], x, y, (a == b) ? P_EMPTY : P_ERROR);
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
        *out_diff_sequence = diffs;
        return 0; // Successo
    }

    for (int k = 0; k < f_a; k++)
    {
        if (diffs[k])
        {
            universe_destroy(diffs[k]);
        }
    }
    free(diffs);

    return FILE_FORMAT_ERROR;
}