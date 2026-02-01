#ifndef CUDA_SIMULATION_CUH
#define CUDA_SIMULATION_CUH

#include <stdio.h>
#include "../../universe.h"

/**
 * Maximum proposals a single cell can generate.
 * WATER has the most: down + 2 diagonals + 2 horizontals + stay = 6
 */
#define MAX_PROPOSALS_PER_CELL 6

/**
 * Maximum particles that can compete for a single destination cell.
 * In worst case, a cell could receive proposals from all 8 neighbors + itself.
 * We use 9 to be safe (3x3 neighborhood).
 */
#define MAX_CANDIDATES_PER_CELL 9

/**
 * CUDA block dimensions (16, 16).
 */
#define BLOCK_SIZE 16

typedef struct {
    short src_x;                // Source cell x-coordinate
    short src_y;                // Source cell y-coordinate
    short dest_x;               // Desired destination x-coordinate
    short dest_y;               // Desired destination y-coordinate
    unsigned char type;         // Particle type (P_SAND, P_WATER, etc.)
    unsigned char preference;   // Movement preference rank (0 = first choice)
    int priority;               // Sequential order priority (lower wins)
    unsigned char is_swap;      // True if this is a SAND-WATER swap operation
    unsigned char swap_with;    // Type to place at source if swap (P_WATER)
} Proposal;

typedef struct {
    Proposal candidates[MAX_CANDIDATES_PER_CELL];   // All proposals wanting this cell
    unsigned char candidate_count;                  // Number of valid candidates
    
    // Resolution results
    unsigned char final_type;                       // Particle type that won this cell
    short winner_src_x;                             // Winner's source x
    short winner_src_y;                             // Winner's source y
    unsigned char resolved;                         // True if a winner has been determined
    unsigned char winner_is_swap;                   // True if winner performed a swap
} CellState;

typedef struct {
    // Grid buffers (double-buffered for input/output)
    unsigned char* d_grid_in;       // Input grid state on device
    unsigned char* d_grid_out;      // Output grid state on device
    int width;                      // Grid width
    int height;                     // Grid height
    
    // Proposal system buffers
    Proposal* d_proposals;          // [width * height * MAX_PROPOSALS_PER_CELL]
    unsigned char* d_prop_counts;   // Number of proposals per source cell
    
    // Destination cell states
    CellState* d_cell_states;       // [width * height]
    
    // Per-particle satisfaction tracking (has this particle found a destination?)
    unsigned char* d_satisfied;     // [width * height]
    
    // Synchronization flag for iterative resolution
    int* d_changed;                 // Flag indicating if any resolution occurred
    
    // Swap handling: tracks cells that need water placed due to swaps
    unsigned char* d_swap_sources;  // [width * height] - marks cells receiving water from swaps
} CudaContext;

CudaContext* cuda_context_create(int width, int height);
void cuda_init_context(CudaContext* ctx);
void cuda_context_destroy(CudaContext* ctx);
void cuda_upload_grid(CudaContext* ctx, Universe* u);
void cuda_download_grid(CudaContext* ctx, Universe* out);

#endif