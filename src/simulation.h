/**
 * @file simulation.h
 * @brief Interface for the simulation logic.
 */
#include "universe.h"


#ifndef SIMULATION_H
#define SIMULATION_H

#ifdef __cplusplus
extern "C" {
#endif

#define SAND_NOISE_CHANCE 0.4f
#define WATER_FALL_DOWN_CHANCE 0.9f
#define WATER_FALL_DENSITY_CHANCE 1.0f
#define WATER_MOVE_DIAGONAL_CHANCE 0.6f
#define WATER_MOVE_HORIZONTAL_CHANCE 0.9f

/**
 * @brief Calculates the next state of the universe based on the current state.
 * @param universe Pointer to the current Universe state.
 * @param out Pointer to the Universe where the next state will be stored.
 * @param generation The current generation index (0-based).
 * @return Pointer to a newly allocated Universe representing the next state.
 * The caller is responsible for destroying the returned Universe.
 */
void next(Universe* universe, Universe* out, int generation);

#ifdef __cplusplus
}
#endif

#endif // SIMULATION_H