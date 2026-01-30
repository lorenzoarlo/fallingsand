/**
 * @file test.h
 * @brief Header file for test functionalities, including comparison of .sand files.
 */
#ifndef TEST_H
#define TEST_H

#include "universe.h" // For Universe type


#define PARAMETERS_NO_VALID_ERROR 1
#define FILE_NULL_ERROR 2
#define PARAMETERS_MISMATCH_ERROR 3
#define MEMORY_FAIL_ERROR 4
#define FILE_FORMAT_ERROR 5

/**
 * Compare two .sand files and generate a sequence of differences universes.
 * @param filename_a First .sand file path
 * @param filename_b Second .sand file path
 * @param out_diff_sequence Pointer to return the array of difference universes
 * @param out_frames Pointer to return the number of frames
 * @return 0 on success, something else > 0 on errors (e.g., different sizes or file not found)
 */
int test_sand(const char *filename_a, const char *filename_b, Universe ***out_diff_sequence, int *out_frames);

#endif // TEST_H