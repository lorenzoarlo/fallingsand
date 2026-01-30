
/**
 * @file io.h
 * @brief Header file for input/output operations related to the falling sand simulation.
 * It defines a new file format ".sand" to store simulation frames.
 */
#ifndef IO_H
#define IO_H

#include "universe.h"
#include <stdio.h> // For FILE type

// Magic number: "SAND" as ASCII bytes
#define FILE_MAGIC "SAND"

#define FRAME_POSITION_OFFSET 12 // Offset where frame data starts in the file

/**
 * @brief Creates a new file for storing the simulation frames.
 * The file starts with a magic number (4 byte) followed by width (4 byte), height (4 byte) and number of frames (4 byte).
 *
 * @param filename The name of the file to create.
 * @param width The width of the universe grid.
 * @param height The height of the universe grid.
 * @return Pointer to the opened FILE.
 */
FILE *io_create_file(const char *filename, int width, int height);

/**
 * @brief Appends a frame of the universe to the file.
 *
 * @param file Pointer to the opened FILE.
 * @param universe Pointer to the Universe whose state is to be saved.
 */
void io_append_frame(FILE *file, Universe *universe);

/**
 * @brief Opens an existing simulation file for reading.
 *
 * @param filename The name of the file to open.
 * @param width Pointer to store the width of the universe grid.
 * @param height Pointer to store the height of the universe grid.
 * @param num_frames Pointer to store the number of frames in the file.
 * @return Pointer to the opened FILE.
 */
FILE *io_open_read(const char *filename, int *width, int *height, int *num_frames);

/**
 * @brief Reads a frame from the file into the given Universe.
 *
 * @param file Pointer to the opened FILE.
 * @param u Pointer to the Universe where the frame will be loaded.
 * @return 0 on success, -1 on failure.
 */
int io_read_frame(FILE *file, Universe *u);

/**
 * @brief Seeks to a specific frame in the file.
 *
 * @param file Pointer to the opened FILE.
 * @param frame_index The index of the frame to seek to.
 * @param width The width of the universe grid.
 * @param height The height of the universe grid.
 * @return 0 on success, -1 on failure.
 */
int io_seek_frame(FILE *file, int frame_index, int width, int height);

#endif
