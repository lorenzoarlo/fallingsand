/**
 * @file main.c
 * @brief Main entry point for the Falling Sand simulation engine.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h> // For mkdir handling if needed (mostly POSIX)
#include <unistd.h>   // For fork(), sysconf()
#include <sys/wait.h> // For wait()

#include "cli.h"
#include "io.h"
#include "universe.h"
#include "graphics.h"
#include "test.h"
#include "simulation.h"
#include "performance.h"

#define SUCCESS 0
#define GENERIC_ERROR 255

/**
 * @brief Parse command-line arguments and open the input file.
 * @param argc Argument count.
 * @param argv Argument vector.
 * @param width Pointer to store the width of the universe.
 * @param height Pointer to store the height of the universe.
 * @param frames Pointer to store the number of frames to simulate.
 * @param input_file Pointer to store the opened input file.
 * @return CLIConfig structure with parsed configuration.
 */
CLIConfig starting(int argc, char *argv[], int *width, int *height, int *frames, FILE **input_file)
{
    // Parsing command-line arguments
    CLIConfig config;
    if (parse_arguments(argc, argv, &config) != CLI_SUCCESS)
    {
        perror("Error parsing command-line arguments");
        exit(GENERIC_ERROR); // Exit with error (help is printed inside parse_arguments)
    }

    printf("Starting simulation...\n");
    printf("Input file: %s\n", config.input_filename);
    printf("Output: %s\n", config.output_filename);
    printf("Frames to simulate: %d\n", config.frames);

    *input_file = io_open_read(config.input_filename, width, height, frames);
    if (!*input_file)
    {
        perror("main_starting -> Error opening input file");
        exit(GENERIC_ERROR);
    }
    return config;
}

/**
 * @brief Setup the initial universe state from the input file.
 * @param width Width of the universe.
 * @param height Height of the universe.
 * @param input_file File pointer to the input .sand file.
 * @return Pointer to the initialized Universe structure.
 */
Universe *setup(int width, int height, FILE *input_file)
{
    Universe *current_universe = universe_create(width, height);
    if (!current_universe)
    {
        perror("Failed to allocate initial universe");
        // Clean resources
        fclose(input_file);
        exit(GENERIC_ERROR);
    }

    // Read only the first frame (initial state)
    if (io_read_frame(input_file, current_universe) != 0)
    {
        perror("main_setup -> Error reading initial frame");
        // Clean resources
        universe_destroy(current_universe);
        fclose(input_file);
        exit(GENERIC_ERROR);
    }
    return current_universe;
}

#define GREEN     "\033[0;32m"
#define RESET     "\033[0m"
/**
 * @brief Run the testing verification against a reference .sand file.
 * @param config CLIConfig structure with test parameters.
 */
void testing(CLIConfig config)
{
    printf("Running verification against %s\n", config.test_filename);

    Universe **diffs_array = NULL;    // Temporary array to hold diffs
    int diff_frames = 0;

    // Pass address of diffs_array pointer
    int result = test_sand(config.output_filename, config.test_filename, &diffs_array, &diff_frames);

    if (result == 0)
    {
        printf(GREEN "Test SUCCESS: The output matches the test file.\n" RESET);
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
                    snprintf(diff_path, sizeof(diff_path), "%s/testdiff_%04d.ppm", config.output_folder, k + 1);
                    universe_export_image(diffs_array[k], diff_path, config.scale);
                }
            }
        }
    }

    // Cleanup diffs only if they were allocated
    if (!diffs_array)
    {
        return;
    }

    for (int k = 0; k < diff_frames; k++)
    {
        if (diffs_array[k])
        {
            universe_destroy(diffs_array[k]);
        }
    }
    free(diffs_array);
}

/**
 * @brief Handles parallel image generation using fork().
 * Compatible with macOS and Unix systems.
 * * @param config The CLI configuration.
 * @param width Universe width.
 * @param height Universe height.
 * @param total_frames Total frames to process.
 */
void produce_images_parallel(CLIConfig config, int width, int height, int total_frames)
{
    if (config.output_folder == NULL)
    {
        return;
    }

    printf("Starting parallel image generation...\n");

    // Get number of available processors
    // _SC_NPROCESSORS_ONLN is standard POSIX
    long num_procs = sysconf(_SC_NPROCESSORS_ONLN);
    if (num_procs < 1)
        num_procs = 1;

    printf("Spawning %ld processes for image generation.\n", num_procs);

    pid_t pid;
    long child_id = -1;

    // Fork loop
    for (long i = 0; i < num_procs; i++)
    {
        pid = fork();
        if (pid < 0)
        {
            perror("Fork failed");
            exit(GENERIC_ERROR);
        }
        else if (pid == 0)
        {
            // We are in the child process
            child_id = i;
            break; // Break the loop so the child doesn't fork more children
        }
    }

    if (child_id != -1)
    {
        // === CHILD PROCESS LOGIC ===

        // Open the generated .sand file for reading
        // We use dummy variables for w, h, f because we already know them
        int w, h, f;
        FILE *fp = io_open_read(config.output_filename, &w, &h, &f);
        if (!fp)
        {
            perror("Child process: error opening .sand file");
            exit(GENERIC_ERROR);
        }

        Universe *u = universe_create(width, height);

        // Iterate through all frames in the file
        // Note: Depending on io implementation, we might need to read sequentially
        // to advance the file pointer correctly.
        for (int i = 0; i < total_frames; i++)
        {
            // Read the frame into universe u
            if (io_read_frame(fp, u) != 0)
                break;

            // Round-robin distribution:
            // Process ONLY if (frame_index % num_procs) matches this child's ID
            if ((i % num_procs) == child_id)
            {
                char image_path[512];
                // i + 1 for user friendly 1-based index
                snprintf(image_path, sizeof(image_path), "%s/%04d.png", config.output_folder, i + 1);
                universe_export_image(u, image_path, config.scale);
            }
        }

        universe_destroy(u);
        fclose(fp);
        exit(SUCCESS); // Child exits here
    }
    else
    {
        // Wait for all children to finish
        int status;
        while (wait(&status) > 0)
            ;
        printf("All images generated.\n");
    }
}

int main(int argc, char *argv[])
{
    int width, height, frames;
    FILE *input_file;
    CLIConfig config;

    config = starting(argc, argv, &width, &height, &frames, &input_file);

    // Create universe for the current state
    Universe *current_universe = setup(width, height, input_file);

    // Close input file, we don't need subsequent frames from input
    fclose(input_file);

    // Prepare output file
    FILE *output_file = io_create_file(config.output_filename, width, height);
    if (!output_file)
    {
        perror("main -> Error creating output file");
        universe_destroy(current_universe);
        exit(GENERIC_ERROR);
    }
    printf("Created output file: %s\n", config.output_filename);

    FILE *performance_file = fopen(config.performance_filemane, "w");
    if (!performance_file)
    {
        perror("main -> Error creating performance log file");
        universe_destroy(current_universe);
        fclose(output_file);
        exit(GENERIC_ERROR);
    }

    uint64_t start, end;
    Universe *next_universe;
    for (int i = 0; i < config.frames; i++)
    {
        // Avoid useless line printing, do it in the same line 
        printf("Simulating frame %d / %d\r", i + 1, config.frames);
        fflush(stdout);

        // Calculate next generation
        start = measure_start();
        Universe *next_universe = next(current_universe, i);
        end = measure_start();
        append_performance(performance_file, start, end, i + 1);
        if (!next_universe)
        {
            fprintf(stderr, "\nError at frame %d\n", i);
            break;
        }

        // Append to .sand file
        io_append_frame(output_file, next_universe);

        // Clean up old state and move to new state
        universe_destroy(current_universe);
        current_universe = next_universe;
    }
    printf("Simulation completed. Data saved.\n");
    fclose(performance_file);

    // Clean up final state and close output file
    universe_destroy(current_universe);
    fclose(output_file);

    // Now that the .sand file is complete, we generate images in parallel
    if (config.output_folder != NULL)
    {
        produce_images_parallel(config, width, height, config.frames);
    }

    // Test verification if test file is provided
    if (config.test_filename != NULL)
    {
        testing(config);
    }

    return 0;
}