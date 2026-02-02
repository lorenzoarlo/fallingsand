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