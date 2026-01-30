/**
 * @file universe.h
 * @brief Defines functions for managing 2D grid of falling sand simulation.
 */
#ifndef UNIVERSE_H
#define UNIVERSE_H

#define P_EMPTY 0
#define P_WALL 1
#define P_SAND 2
#define P_WATER 3
#define P_ERROR 255

/**
 * @brief Macro to calculate the index in the cells array for given (x, y) coordinates.
 */
#define UINDEX(x, y, width) ((y) * (width) + (x))

/**
 * Universe structure representing a 2D grid of cells.
 */
typedef struct
{
    unsigned char *cells;
    int width;
    int height;
} Universe;

/**
 * @brief Creates a new Universe with specified width and height.
 *
 * @param width The width of the universe grid.
 * @param height The height of the universe grid.
 * @return Pointer to the newly created Universe.
 */
Universe *universe_create(int width, int height);

/**
 * @brief Destroys the given Universe and frees associated memory.
 *
 * @param universe Pointer to the Universe to be destroyed.
 */
void universe_destroy(Universe *universe);

/**
 * @brief Checks if the given coordinates are out of bounds in the Universe.
 *
 * @param universe Pointer to the Universe.
 * @param x The x-coordinate.
 * @param y The y-coordinate.
 * @return 1 if out of bounds, 0 otherwise.
 */
int universe_out_of_bounds(Universe *universe, int x, int y);

/**
 * @brief Gets the particle type at the specified coordinates in the Universe.
 *
 * @param universe Pointer to the Universe.
 * @param x The x-coordinate.
 * @param y The y-coordinate.
 * @return The particle type at the specified coordinates.
 */
unsigned char universe_get(Universe *universe, int x, int y);

/**
 * @brief Set the specified coordinates in the Universe as a cell.
 *
 * @param universe Pointer to the Universe.
 * @param x The x-coordinate.
 * @param y The y-coordinate.
 * @param value The particle type to set at the specified coordinates.
 * @return The particle type at the specified coordinates.
 */
void universe_set(Universe *universe, int x, int y, unsigned char value);

#endif
