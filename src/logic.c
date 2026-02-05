#include "simulation.h"
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

#include "utility/utility-functions.h"


/**
 * Compute the next generation of the universe
 */
void next(Universe *u, Universe *out, int generation)
{

    // Copying by default the entire universe
    memcpy(out->cells, u->cells, u->width * u->height);

    // Choose which cell of the 2x2 block to offset based on generation
    // Pattern Margolus for alternates (0,0) -> (1,1) -> (0,1) -> (1,0)
    int phase = generation % 4;
    int offset_x = (phase == 1 || phase == 3) ? 1 : 0;
    int offset_y = (phase == 1 || phase == 2) ? 1 : 0;

    // Iterate over the universe in 2x2 blocks
    for (int y = offset_y; y < u->height - 1; y += 2)
    {
        for (int x = offset_x; x < u->width - 1; x += 2)
        {
            blocklogic(x, y, u, out->cells, generation);
        }
    }
}