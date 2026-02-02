#include "kernels.cuh"
__device__ inline unsigned char wrap_get(
    const unsigned char* in,
    const unsigned char* out,
    const unsigned char* clock,
    int idx)
{
    return clock[idx] ? out[idx] : in[idx];
}

__device__ inline void mark_updated(unsigned char* clock, int idx)
{
    clock[idx] = 1;
}

__global__ void row_kernel(
    const unsigned char* in,
    unsigned char* out,
    unsigned char* clock,
    int w, int h,
    int y,
    int start_x,
    int step_x,
    int gen)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= w) return;

    int x = start_x + i * step_x;
    int idx = y * w + x;

    if (clock[idx]) return;

    unsigned char me = in[idx];
    out[idx] = me;
    mark_updated(clock, idx);

    bool left_first = (gen % 2 == 0);

    // ============ SAND ============
    if (me == P_SAND) {
        int below_y = y + 1;
        if (below_y < h) {
            int below = below_y * w + x;
            unsigned char cell_below = wrap_get(in, out, clock, below);

            if (cell_below == P_EMPTY) {
                out[below] = P_SAND;
                out[idx] = P_EMPTY;
                mark_updated(clock, below);
                return;
            }

            int dir = left_first ? -1 : 1;
            int d1x = x + dir;
            int d2x = x - dir;

            if (d1x >= 0 && d1x < w) {
                int d1 = below_y * w + d1x;
                if (wrap_get(in, out, clock, d1) == P_EMPTY) {
                    out[d1] = P_SAND;
                    out[idx] = P_EMPTY;
                    mark_updated(clock, d1);
                    return;
                }
            }

            if (d2x >= 0 && d2x < w) {
                int d2 = below_y * w + d2x;
                if (wrap_get(in, out, clock, d2) == P_EMPTY) {
                    out[d2] = P_SAND;
                    out[idx] = P_EMPTY;
                    mark_updated(clock, d2);
                    return;
                }
            }

            if (cell_below == P_WATER) {
                if ((x + y + gen) % 2 == 0) {
                    out[below] = P_SAND;
                    out[idx] = P_WATER;
                    mark_updated(clock, below);
                }
            }
        }
        return;
    }

    // ============ WATER ============
    if (me == P_WATER) {
        int below_y = y + 1;
        if (below_y < h) {
            int below = below_y * w + x;
            unsigned char cell_below = wrap_get(in, out, clock, below);

            if (cell_below == P_EMPTY) {
                out[below] = P_WATER;
                out[idx] = P_EMPTY;
                mark_updated(clock, below);
                return;
            }

            int left_x = x - 1;
            int right_x = x + 1;

            int first_x = left_first ? left_x : right_x;
            int second_x = left_first ? right_x : left_x;

            if (first_x >= 0 && first_x < w) {
                int d = below_y * w + first_x;
                if (wrap_get(in, out, clock, d) == P_EMPTY) {
                    out[d] = P_WATER;
                    out[idx] = P_EMPTY;
                    mark_updated(clock, d);
                    return;
                }
            }

            if (second_x >= 0 && second_x < w) {
                int d = below_y * w + second_x;
                if (wrap_get(in, out, clock, d) == P_EMPTY) {
                    out[d] = P_WATER;
                    out[idx] = P_EMPTY;
                    mark_updated(clock, d);
                    return;
                }
            }

            int h_first = left_first ? right_x : left_x;
            int h_second = left_first ? left_x : right_x;

            if (h_first >= 0 && h_first < w) {
                int hidx = y * w + h_first;
                if (wrap_get(in, out, clock, hidx) == P_EMPTY) {
                    out[hidx] = P_WATER;
                    out[idx] = P_EMPTY;
                    mark_updated(clock, hidx);
                    return;
                }
            }

            if (h_second >= 0 && h_second < w) {
                int hidx = y * w + h_second;
                if (wrap_get(in, out, clock, hidx) == P_EMPTY) {
                    out[hidx] = P_WATER;
                    out[idx] = P_EMPTY;
                    mark_updated(clock, hidx);
                }
            }
        }
    }
}
