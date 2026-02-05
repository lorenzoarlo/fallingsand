#include <hwy/highway.h>
#include "../simulation.h"
#include "../utility/utility-functions.h" // For probability constant and definitions

// Standard namespace alias for Highway
namespace hn = hwy::HWY_NAMESPACE;

/**
 * Conditional swap function using a mask
 * if bit in condition_mask is set, swap a and b at that position
 * @template Descriptor - Highway descriptor. Needed by highway to deduce automatically data size and lanes
 * @template Vector - Vector type of the data that we want to swap.
 * @template Mask - Mask type
 * @param d - Descriptor
 * @param condition_mask - Mask indicating where to swap [1010] means swap elements 1 and 3
 * @param a - First vector (will be modified)
 * @param b - Second vector (will be modified)
 */
template <class Descriptor, class Vector, class Mask>
HWY_INLINE void IfSwap(Descriptor d, Mask condition_mask, Vector &a, Vector &b)
{
    auto new_a = hn::IfThenElse(condition_mask, b, a);
    auto new_b = hn::IfThenElse(condition_mask, a, b);
    a = new_a;
    b = new_b;
}

/**
 * Generates a random mask based on probability.
 * Since random_hash is scalar, we compute it per lane and pack results.
 */
template <class D>
HWY_INLINE hn::Mask<D> RandomMask(D d, size_t x, size_t y, size_t lanes, float probability, int generation, int salt)
{
    // We allocate statically to avoid dynamic allocation in a loop.
    // HWY_MAX_BYTES is the maximum number of bytes a vector can have in Highway.
    bool check[HWY_MAX_BYTES];

    // Fill only the needed lanes (the other are ignored)
    for (size_t i = 0; i < lanes; i++)
    {
        check[i] = random_hash(x + i * 2, y, generation, salt) < probability;
    }
    // Transform to vector
    const uint8_t *ptr = reinterpret_cast<const uint8_t *>(check);
    // Load into a vector
    auto vec_bools = hn::Load(d, ptr);

    return hn::Gt(vec_bools, hn::Zero(d));
}

/**
 * Main logic for every "block-row".
 * @param d - Highway descriptor
 * @param toprow_address - Pointer to the first pixel of the top row of the block
 * @param bottomrow_address - Pointer to the first pixel of the bottom row of the block
 * @param floorrow_address - Pointer to the first pixel of the row below the block (for floor check). Can be nullptr if out of bounds
 * @param width - Effective width to process
 * @param offset_x - Current column index (for make absolute x in randomness)
 * @param y - Current row index (for randomness)
 * @param lanes - Number of lanes in the vector
 */
void inline Logic(const hn::ScalableTag<uint8_t> d,
                  uint8_t *toprow_address,
                  uint8_t *bottomrow_address,
                  uint8_t *floorrow_address,
                  size_t width, size_t offset_x, size_t y, size_t lanes, int generation)
{
    // alias for Vec type of d
    using V = hn::Vec<decltype(d)>;
    // Vectorized constant
    const auto SAND = hn::Set(d, P_SAND);
    const auto WATER = hn::Set(d, P_WATER);
    const auto WALL = hn::Set(d, P_WALL);

    for (size_t x = 0; x <= width - (2 * lanes); x += (2 * lanes))
    {
        size_t absolute_x = x + offset_x;

        // first lane
        auto toprow = hn::LoadU(d, toprow_address + x);
        auto bottomrow = hn::LoadU(d, bottomrow_address + x);

        // second lane
        auto toprow2nd = hn::LoadU(d, toprow_address + x + lanes);
        auto bottomrow2nd = hn::LoadU(d, bottomrow_address + x + lanes);

        // verify if equals
        auto rows_equal = hn::Eq(toprow, bottomrow);
        auto rows2nd_equal = hn::Eq(toprow2nd, bottomrow2nd);

        // verify if couple blocks are equals
        auto toprow_swapped = hn::Shuffle01(toprow);
        auto toprow2nd_swapped = hn::Shuffle01(toprow2nd);

        // Verify if [a0, a1, ...] = [a1, a0, ...]
        auto shuffledequal = hn::Eq(toprow, toprow_swapped);
        auto shuffledequal2nd = hn::Eq(toprow2nd, toprow2nd_swapped);

        auto block_equals = hn::And(rows_equal, hn::And(rows2nd_equal, hn::And(shuffledequal, shuffledequal2nd)));

        // If all cells are the same, skip processing
        if (hn::AllTrue(d, block_equals))
        {
            continue;
        }

        // Loading interleaved data for the 2 rows. We get 4 vectors:
        // TL, TR, BL, BR
        // TL = toprow[0], toprow[2], toprow[4], ...
        // TR = toprow[1], toprow[3], toprow[5],
        V toplefts, toprights, bottomlefts, bottomrights;

        // Before toprow2nd because inserts before the 3rd argument 
        toplefts = hn::ConcatEven(d, toprow2nd, toprow);
        toprights = hn::ConcatOdd(d, toprow2nd, toprow);
        bottomlefts = hn::ConcatEven(d, bottomrow2nd, bottomrow);
        bottomrights = hn::ConcatOdd(d, bottomrow2nd, bottomrow);

        // Compare every topleft and toprights to sand (to understand if they are sand and they can swap)
        auto topleft_is_sand = hn::Eq(toplefts, SAND);
        auto topright_lighter = hn::Lt(toprights, SAND);

        auto topright_is_sand = hn::Eq(toprights, SAND);
        auto topleft_lighter = hn::Lt(toplefts, SAND);

        // Final mask for top horizontal sand movement
        auto topcanmove = hn::Or(
            hn::And(topleft_is_sand, topright_lighter),
            hn::And(topright_is_sand, topleft_lighter));

        // Check if below is lighter for sand to fall
        auto bottomleft_lighter = hn::Lt(bottomlefts, SAND);
        auto bottomright_lighter = hn::Lt(bottomrights, SAND);

        auto bottom_lighter = hn::And(bottomleft_lighter, bottomright_lighter);

        // Horizontal swap can happens only if below is lighter (it is falling)
        auto sand_horizontal_move = hn::And(topcanmove, bottom_lighter);
        auto chance_mask = RandomMask(d, absolute_x, y, lanes, SAND_NOISE_CHANCE, generation, 0); // salt is 0 as the original

        auto topcanswaphorizontally = hn::And(sand_horizontal_move, chance_mask);
        // Swap if both conditions are met
        IfSwap(d, topcanswaphorizontally, toplefts, toprights);

        // if (!top_moved && *topleft == P_SAND)
        //     {
        //         // Can fall down?
        //         if (*bottomleft < P_SAND)
        //         {
        //             if (random_hash(x, y, generation, 1) < WATER_FALL_DOWN_CHANCE)
        //             {
        //                 swap(topleft, bottomleft);
        //             }
        //         }
        //         // Can fall diagonally?
        //         else if (*topright < P_SAND && *bottomright < P_SAND)
        //         {
        //             swap(topleft, bottomright);
        //         }
        //     }
        // In quali casi dobbiamo scambiare?
        // La prima condizione è che topleft sia SAND e che non si sia mosso.
        // A questo punto abbiamo due possibilità:
        // - bottomleft è più leggero di topleft e randomhash < WATER_FALL_DOWN_CHANCHE (può cadere dritto)
        // - bottomleft è più pesante di topleft, topright è più leggero della sabbia e  bottomright è più leggero della sabbia (può cadere diagonalmente)

        // To verify that sand has not already swapped horizontally
        auto sand_not_already_swapped = hn::Not(topcanswaphorizontally); // if it has moved horizontally, it cannot fall vertically

        // RandomMask for falling
        auto randommask_fall = RandomMask(d, absolute_x, y, lanes, WATER_FALL_DOWN_CHANCE, generation, 1);

        // Topleft sand fall, if bottomleft_lighter and randommask_fall, it can fall vertically.
        auto topleftsand_can_fall_vertical = hn::And(topleft_is_sand, hn::And(bottomleft_lighter, randommask_fall));
        // If not is lighter, and top right and bottom right are lighter, it can fall diagonally
        auto topleftdiagonally = hn::And(hn::Not(bottomleft_lighter), hn::And(topright_lighter, bottomright_lighter));
        auto topleftsand_can_fall_diagonal = hn::And(topleft_is_sand, topleftdiagonally);
        // Swap vertical
        IfSwap(d, hn::And(sand_not_already_swapped, topleftsand_can_fall_vertical),
               toplefts, bottomlefts);
        // Swap diagonal
        IfSwap(d, hn::And(sand_not_already_swapped, topleftsand_can_fall_diagonal),
               toplefts, bottomrights);

        // Logic is the same for topleft sand, but mirrored

        // topright sand fall, if bottomright_lighter and randommask_fall, it can fall vertically.
        auto toprightsand_can_fall_vertical = hn::And(topright_is_sand, hn::And(bottomright_lighter, randommask_fall));
        // If not is lighter, and top left and bottom left are lighter, it can fall diagonally
        auto toprightdiagonally = hn::And(hn::Not(bottomright_lighter), hn::And(topleft_lighter, bottomleft_lighter));
        auto toprightsand_can_fall_diagonal = hn::And(topright_is_sand, toprightdiagonally);
        // Swap vertical
        IfSwap(d, hn::And(sand_not_already_swapped, toprightsand_can_fall_vertical),
               toprights, bottomrights);
        // Swap diagonal
        IfSwap(d, hn::And(sand_not_already_swapped, toprightsand_can_fall_diagonal),
               toprights, bottomlefts);

        // Manage WATER falling and horizontal movement

        auto randommask_watermove_vertical = RandomMask(d, absolute_x, y, lanes, WATER_FALL_DENSITY_CHANCE, generation, 2);
        auto randommask_watermove_diagonal = RandomMask(d, absolute_x, y, lanes, WATER_MOVE_DIAGONAL_CHANCE, generation, 3);
        // Manage TOPLEFT if WATER
        auto topleft_is_water = hn::Eq(toplefts, WATER);

        // if (*topleft == P_WATER)
        // {
        //     // Is the cell below lower density?
        //     if (*bottomleft < *topleft && random_hash(x, y, generation, 2) < WATER_FALL_DENSITY_CHANCE)
        //     {
        //         swap(topleft, bottomleft);
        //         drop_left = 1;
        //     }
        //     // Can move diagonally?
        //     else if (*topright < *topleft && *bottomright < *topleft && random_hash(x, y, generation, 3) < WATER_MOVE_DIAGONAL_CHANCE)
        //     {
        //         swap(topleft, bottomright);
        //         drop_left = 1;
        //     }
        // }
        // L'acqua cade se:
        // - verticalmente, la cella sotto è di densità inferiore e randomhash < WATER_FALL_DENSITY_CHANCE
        // - diagonalmente se NON è caduta verticalmente, se topright e bottomright sono di densità inferiore e randomhash < WATER_MOVE_DIAGONAL_CHANCE

        auto waterbelow_lighter_left = hn::Lt(bottomlefts, toplefts);
        auto topleftwater_can_fall_vertical = hn::And(topleft_is_water, hn::And(waterbelow_lighter_left, randommask_watermove_vertical));

        // look if the right cells are lighter for diagonal movement
        auto waterdiagonal_lighter_left = hn::And(hn::Lt(toprights, toplefts), hn::Lt(bottomrights, toplefts));
        auto topleftwater_can_fall_diagonal = hn::And(topleft_is_water, hn::And(waterdiagonal_lighter_left, randommask_watermove_diagonal));

        // Swap vertical
        IfSwap(d, topleftwater_can_fall_vertical, toplefts, bottomlefts);
        // Swap diagonal
        IfSwap(d, topleftwater_can_fall_diagonal, toplefts, bottomrights);

        // Manage TOPRIGHT if WATER
        auto topright_is_water = hn::Eq(toprights, WATER);
        // Logic is the same as topleft, but mirrored
        auto waterbelow_lighter_right = hn::Lt(bottomrights, toprights);
        auto toprightwater_can_fall_vertical = hn::And(topright_is_water, hn::And(waterbelow_lighter_right, randommask_watermove_vertical));
        // look if the left cells are lighter for diagonal movement
        auto waterdiagonal_lighter_right = hn::And(hn::Lt(toplefts, toprights), hn::Lt(bottomlefts, toprights));
        auto toprightwater_can_fall_diagonal = hn::And(topright_is_water, hn::And(waterdiagonal_lighter_right, randommask_watermove_diagonal));
        // Swap vertical
        IfSwap(d, toprightwater_can_fall_vertical, toprights, bottomrights);
        // Swap diagonal
        IfSwap(d, toprightwater_can_fall_diagonal, toprights, bottomlefts);

        // Verify if water has dropped (either left or right)
        auto water_dropped = hn::Or(
            hn::Or(topleftwater_can_fall_vertical, topleftwater_can_fall_diagonal),
            hn::Or(toprightwater_can_fall_vertical, toprightwater_can_fall_diagonal));

        // Make horizontal movement only if water has not dropped
        auto water_not_dropped = hn::Not(water_dropped);

        auto topleft_lighter_than_water = hn::Lt(toplefts, WATER);
        auto topright_lighter_than_water = hn::Lt(toprights, WATER);

        // Top can move horizontally if one is water and the other is lighter
        auto top_can_move_horizontally = hn::Or(
            hn::And(topleft_is_water, topright_lighter_than_water),
            hn::And(topright_is_water, topleft_lighter_than_water));

        // Below is solid if both bottom cells are >= WATER (water or heavier)
        auto bottomleft_solid = hn::Ge(bottomlefts, WATER);
        auto bottomright_solid = hn::Ge(bottomrights, WATER);
        auto below_solid = hn::And(bottomleft_solid, bottomright_solid);

        // Random chance for horizontal movement
        auto randommask_water_horizontal = RandomMask(d, absolute_x, y, lanes, WATER_MOVE_HORIZONTAL_CHANCE, generation, 4);

        // Final condition: can move, not dropped, and (below solid OR random chance)
        auto top_water_horizontal_swap = hn::And(
            hn::And(water_not_dropped, top_can_move_horizontally),
            hn::Or(below_solid, randommask_water_horizontal));

        // Perform the swap
        IfSwap(d, top_water_horizontal_swap, toplefts, toprights);

        // Look if there is solid floor below
        V floor_lefts, floor_rights;
        if (floorrow_address != nullptr)
        {
            hn::LoadInterleaved2(d, floorrow_address + x, floor_lefts, floor_rights);
        }
        else
        {
            floor_lefts = WALL;
            floor_rights = WALL;
        }

        auto floor_left_solid = hn::Ge(floor_lefts, WATER);
        auto floor_right_solid = hn::Ge(floor_rights, WATER);
        auto floor_solid = hn::And(floor_left_solid, floor_right_solid);

        auto bottomwater_canmove_horizontally = hn::Or(
            hn::And(hn::Eq(bottomlefts, WATER), hn::Lt(bottomrights, WATER)),
            hn::And(hn::Eq(bottomrights, WATER), hn::Lt(bottomlefts, WATER)));

        IfSwap(d,
               hn::And(bottomwater_canmove_horizontally,
                       hn::Or(floor_solid,
                              RandomMask(d, absolute_x, y, lanes, WATER_MOVE_HORIZONTAL_CHANCE, generation, 5))),
               bottomlefts, bottomrights);

        // Write back the results
        hn::StoreInterleaved2(toplefts, toprights, d, toprow_address + x);
        hn::StoreInterleaved2(bottomlefts, bottomrights, d, bottomrow_address + x);
    }
}

/**
 * Next simulation step using SIMD optimizations
 * Externalized to make it compatible
 */
extern "C" void next(Universe *u, Universe *out, int generation)
{
    // Copying by default the entire universe
    memcpy(out->cells, u->cells, u->width * u->height);

    int phase = generation % 4;
    // if 00 -> offset (0,0)
    // if 01 -> offset (1,1)
    // if 10 -> offset (0,1)
    // if 11 -> offset (1,0)
    int offset_x = phase & 1;                  // x is always equal to the least significant bit
    int offset_y = (phase >> 1) ^ (phase & 1); // y is xor of both bits

    // Highway setup
    const hn::ScalableTag<uint8_t> d; // Descriptor for uint8_t vectors. It automatically deduces size of .
    // a lane is the number of elements of type uint8_t in a vector (based on the architecture used)
    const size_t lanes = hn::Lanes(d); // Automatically deduce number of lanes

    // Iterate over the rows in 2x2 blocks
    for (int y = offset_y; y < u->height - 1; y += 2)
    {
        // (u->width - 1) because we read pairs of pixels. - offset_x to adjust for phase
        int real_width = u->width - 1 - offset_x;

        // Address for the two rows
        unsigned char *row_top = &out->cells[(y * u->width) + offset_x];
        unsigned char *row_bottom = &out->cells[((y + 1) * u->width) + offset_x];

        unsigned char *row_floor = (y + 2 < u->height)
                                       ? &out->cells[((y + 2) * u->width) + offset_x]
                                       : nullptr;

        Logic(d, row_top, row_bottom, row_floor, real_width, offset_x, y, lanes, generation);

        // Process the remaining columns with scalar code (if width is not a multiple of 2*lanes)
        size_t processed = (real_width / (2 * lanes)) * (2 * lanes);
        for (size_t x = offset_x + processed; x < u->width - 1; x += 2)
        {
            blocklogic(x, y, u, out->cells, generation);
        }
    }
}