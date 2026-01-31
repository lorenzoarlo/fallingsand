#include "cli.h"

void cli_help(const char *prog_name)
{
    printf("Usage: %s <input_file> <output_file> <frames> [options]\n\n", prog_name);
    printf("Parameters:\n");
    printf("\tinput_file     : Path for initial state file. It should be encoded in .sand format\n");
    printf("\toutput_file    : Path for output .sand file to store simulation frames\n");
    printf("\tframes         : Number of frames to simulate (positive integer)\n");
    printf("Options:\n");
    printf("\t -s <scale>    : (Optional) Scale factor for exported images (default: 1)\n");
    printf("\t-oi <folder>    : (Optional) Folder to save output images for each frame\n");
    printf("\t-t <file>      : (Optional) Path for test file\n");
    printf("\t-l <logfile>  : (Optional) Path for performance log file (default: performance.log)\n");
}

int parse_arguments(int argc, char *argv[], CLIConfig *config)
{
    // Help requested
    if (argc == 2 && strcmp(argv[1], "help") == 0)
    {
        cli_help(argv[0]);
        return CLI_SUCCESS; // Returns success because we don't need to start the simulation
    }

    // Validate number of mandatory arguments
    // Must have at least program name + 3 positional args
    if (argc < 4)
    {
        perror("Error: Invalid number of arguments. Missing mandatory parameters");
        cli_help(argv[0]);
        return CLI_FAILURE_PARAMETER_MISSING;
    }

    // Parse mandatory positional arguments
    config->input_filename = argv[1];
    config->output_filename = argv[2];
    config->frames = atoi(argv[3]);

    if (config->frames <= 0)
    {
        perror("Error, frames must be a positive integer.\n");
        return CLI_FAILURE_INVALID_VALUE;
    }

    // Initialize optional arguments to NULL
    config->output_folder = NULL;
    config->test_filename = NULL;

    // Parse optional arguments with flags
    for (int i = 4; i < argc; i++)
    {
        if (strcmp(argv[i], "-oi") == 0)
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "Error, option -oi requires a folder path.\n");
                return CLI_FAILURE_PARAMETER_MISSING;
            }
            // Next argument is the output folder
            config->output_folder = argv[++i];
        }
        else if (strcmp(argv[i], "-t") == 0)
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "Error, option -t requires a file path.\n");
                return CLI_FAILURE_PARAMETER_MISSING;
            }
            config->test_filename = argv[++i];
        }
        else if (strcmp(argv[i], "-s") == 0)
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "Error: Option -s requires a scale factor.\n");
                return CLI_FAILURE_PARAMETER_MISSING;
            }
            config->scale = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-l") == 0)
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "Error: Option -l requires a scale factor.\n");
                return CLI_FAILURE_PARAMETER_MISSING;
            }
            config->performance_filename = argv[++i];
        }
        else
        {
            fprintf(stderr, "Error, unknown argument '%s'.\n", argv[i]);
            cli_help(argv[0]);
            return CLI_FAILURE_UNKNOWN_PARAMETER;
        }
    }

    // Successful parsing
    return CLI_SUCCESS;
}