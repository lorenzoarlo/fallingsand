#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * @file cli.h
 * @brief Header file for command-line interface parsing for the falling sand simulation.
 */
#ifndef CLI_H
#define CLI_H
typedef struct
{
    char *input_filename;  // Initial state. Should be in .sand format. It will start from the last written file.
    char *output_filename; // Name of the output file
    int frames;            // Number of frames to simulate
    char *test_filename;   // Path for test file (optional)
    char *output_folder;   // Folder to save output images (optional)
    int scale;             // Scale factor for exported images (optional)
} CLIConfig;

/**
 * @brief Parses command-line arguments and fills the CLIConfig structure.
 * @param argc Argument count.
 * @param argv Argument vector.
 * @param config Pointer to CLIConfig structure to be filled.
 * @return true if parsing was successful, false otherwise.
 */
int parse_arguments(int argc, char *argv[], CLIConfig *config);

/**
 * @brief Displays help information for using the command-line interface.
 * @param prog_name Name of the program (usually argv[0]).
 */
void cli_help(const char *prog_name);

#endif // CLI_H