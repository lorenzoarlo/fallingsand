#include "cli.h"


void cli_help(const char *prog_name)
{
    printf("Usage: %s <input_file> <output_file> <frames> [options]\n\n", prog_name);
    printf("Parameters:\n");
    printf("\tinput_file     : Path for initial state file. It should be encoded in .sand format\n");
    printf("\toutput_file    : Path for output .sand file to store simulation frames\n");
    printf("\tframes         : Number of frames to simulate (positive integer)\n");
    printf("Options:\n");
    printf("\t-oi <folder>    : (Optional) Folder to save output images for each frame\n");
    printf("\t-t <file>      : (Optional) Path for test file\n");
}

int parse_arguments(int argc, char *argv[], CLIConfig *config)
{
    // Help requested
    if (argc == 2 && strcmp(argv[1], "help") == 0)
    {
        cli_help(argv[0]);
        return 0; // Returns 0 because we don't need to start the simulation
    }

    // Validate number of mandatory arguments
    // Must have at least program name + 3 positional args
    if (argc < 4)
    {
        perror("Error: Invalid number of arguments. Missing mandatory parameters.\n");
        cli_help(argv[0]);
        return 0;
    }

    // Parse mandatory positional arguments
    config->input_filename = argv[1];
    config->output_filename = argv[2];
    config->frames = atoi(argv[3]);

    if (config->frames <= 0)
    {
        perror("Error: Frames must be a positive integer.\n");
        return 0;
    }

    // Initialize optional arguments to NULL
    config->output_folder = NULL;
    config->test_filename = NULL; // Requires 'test_filename' in CLIConfig struct

    // Parse optional arguments with flags
    for (int i = 4; i < argc; i++)
    {
        if (strcmp(argv[i], "-oi") == 0)
        {
            if (i + 1 < argc)
            {
                config->output_folder = argv[++i];
            }
            else
            {
                fprintf(stderr, "Error: Option -i requires a folder path.\n");
                return 0;
            }
        }
        else if (strcmp(argv[i], "-t") == 0)
        {
            if (i + 1 < argc)
            {
                config->test_filename = argv[++i];
            }
            else
            {
                fprintf(stderr, "Error: Option -t requires a file path.\n");
                return 0;
            }
        }
        else
        {
            fprintf(stderr, "Error: Unknown argument '%s'.\n", argv[i]);
            cli_help(argv[0]);
            return 0;
        }
    }

    // Successful parsing
    return 1;
}