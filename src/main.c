/**
 * @file main.c
 * @brief Main entry point for the Falling Sand simulation engine.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h> // For mkdir handling if needed (mostly POSIX)

#include "cli.h"
#include "io.h"
#include "universe.h"
#include "graphics.h"
#include "test.h"
#include "simulation.h"

int main(int argc, char *argv[])
{
    // Parsing command-line arguments
    CLIConfig config;
    if (!parse_arguments(argc, argv, &config))
    {
        return 1; // Exit with error (help is printed inside parse_arguments)
    }

    printf("Starting simulation...\n");
    printf("Input file: %s\n", config.input_filename);
    printf("Output: %s\n", config.output_filename);
    printf("Frames to simulate: %d\n", config.frames);

    // Initial state
    int width, height, initial_frames;
    FILE *input_file = io_open_read(config.input_filename, &width, &height, &initial_frames);
    if (!input_file)
    {
        perror("Error opening input file");
        return 1;
    }

    // Create universe for the current state
    Universe *current_universe = universe_create(width, height);
    if (!current_universe)
    {
        perror("Failed to allocate initial universe");
        fclose(input_file);
        return 1;
    }

    // Read only the first frame (initial state)
    if (io_read_frame(input_file, current_universe) != 0)
    {
        perror("Error reading initial frame");
        universe_destroy(current_universe);
        fclose(input_file);
        return 1;
    }

    // Close input file, we don't need subsequent frames from input
    fclose(input_file);

    // Prepare output file
    FILE *output_file = io_create_file(config.output_filename, width, height);
    if (!output_file)
    {
        perror("Error creating output file");
        universe_destroy(current_universe);
        return 1;
    }

    // Simulation Loop
    for (int i = 0; i < config.frames; i++)
    {
        // Calculate next generation
        Universe *next_universe = next(current_universe, i);

        if (!next_universe)
        {
            fprintf(stderr, "Error at frame %d\n", i);
            break;
        }

        // Append to .sand file
        io_append_frame(output_file, next_universe);

        // Clean up old state and move to new state
        universe_destroy(current_universe);
        current_universe = next_universe;
    }

    // Export image if requested
    if (config.output_folder != NULL)
    {
        // Open the output .sand file to read frames for image export
        // Reuse same variables for width, height, frames
        FILE *file = io_open_read(config.output_filename, &width, &height, &initial_frames);
        for (int i = 0; i < config.frames; i++)
        {
            Universe *u;
            char image_path[512];
            // Create the image path string: folder/0000.ppm
            // 0 based because the first should be the original one
            snprintf(image_path, sizeof(image_path), "%s/%04d.ppm", config.output_folder, i + 1);
            io_read_frame(file, u);
            universe_export_image(u, image_path, config.scale);
            universe_destroy(u);
        }
        fclose(file);
    }

    // Clean up final state and close output file
    universe_destroy(current_universe);
    fclose(output_file);

    printf("Simulation completed.\n");

    // Test verification if test file is provided
    if (config.test_filename != NULL)
    {
        printf("Running verification against %s\n", config.test_filename);

        Universe ***diff_sequence = NULL; // Will point to array of Universe pointers
        int diff_frames = 0;
        Universe **diffs_array = NULL;

        // Pass address of diffs_array pointer
        int result = test_sand(config.output_filename, config.test_filename, &diffs_array, &diff_frames);

        if (result == 0)
        {
            printf("Test SUCCESS: The output matches the test file.\n");
        }
        else
        {
            fprintf(stderr, "Test FAILED (code %d). Output differs from reference.\n", result);

            // If output folder is available, save difference images
            if (config.output_folder != NULL && diffs_array != NULL)
            {
                printf("Saving difference images to %s...\n", config.output_folder);
                for (int k = 0; k < diff_frames; k++)
                {
                    if (diffs_array[k]) // If there is a diff universe (some implementation might leave it NULL if equal)
                    {
                        char diff_path[512];
                        snprintf(diff_path, sizeof(diff_path), "%s/diff_%04d.ppm", config.output_folder, k + 1);
                        universe_export_image(diffs_array[k], diff_path, config.scale);
                    }
                }
            }
        }

        // Cleanup diffs
        if (diffs_array)
        {
            for (int k = 0; k < diff_frames; k++)
            {
                if (diffs_array[k])
                    universe_destroy(diffs_array[k]);
            }
            free(diffs_array);
        }
    }

    return 0;
}