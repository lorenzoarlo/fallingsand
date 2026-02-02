#include "kernels.cuh"
// Checkerboard helper
__device__ inline bool active_cell(int x, int y, int gen) {
    return ((x + y + gen) & 1) == 0;
}

__device__ unsigned char wrap_get(
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

/*__global__ void sand_step_kernel(
    const unsigned char* in,
    unsigned char* out,
    int w, int h,
    int gen, 
    unsigned char* clock)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;
    if (!active_cell(x, y, gen)) return;

    int idx = y * w + x;
    unsigned char me = in[idx];

    out[idx] = me;

    if (me == P_WALL) return;

    bool left_first = (gen & 1);

    // =====================================================
    // CASE 1: EMPTY → può ricevere SAND o WATER
    // =====================================================
    if (me == P_EMPTY) {

        int up    = idx - w;
        int up_l  = up - 1;
        int up_r  = up + 1;

        // ---- SAND PRIORITY ----
        if (y > 0 && in[up] == P_SAND) {
            out[idx] = P_SAND;
            out[up]  = P_EMPTY;
            return;
        }

        if (y > 0) {
            if (left_first) {
                if (x > 0 && in[up_l] == P_SAND) {
                    out[idx] = P_SAND;
                    out[up_l] = P_EMPTY;
                    return;
                }
                if (x < w-1 && in[up_r] == P_SAND) {
                    out[idx] = P_SAND;
                    out[up_r] = P_EMPTY;
                    return;
                }
            } else {
                if (x < w-1 && in[up_r] == P_SAND) {
                    out[idx] = P_SAND;
                    out[up_r] = P_EMPTY;
                    return;
                }
                if (x > 0 && in[up_l] == P_SAND) {
                    out[idx] = P_SAND;
                    out[up_l] = P_EMPTY;
                    return;
                }
            }
        }

        // ---- WATER ----
        if (y > 0 && in[up] == P_WATER) {
            out[idx] = P_WATER;
            out[up]  = P_EMPTY;
            return;
        }

        if (y > 0) {
            if (left_first) {
                if (x > 0 && in[up_l] == P_WATER) {
                    out[idx] = P_WATER;
                    out[up_l] = P_EMPTY;
                    return;
                }
                if (x < w-1 && in[up_r] == P_WATER) {
                    out[idx] = P_WATER;
                    out[up_r] = P_EMPTY;
                    return;
                }
            } else {
                if (x < w-1 && in[up_r] == P_WATER) {
                    out[idx] = P_WATER;
                    out[up_r] = P_EMPTY;
                    return;
                }
                if (x > 0 && in[up_l] == P_WATER) {
                    out[idx] = P_WATER;
                    out[up_l] = P_EMPTY;
                    return;
                }
            }
        }
    }

    // =====================================================
    // CASE 2: SAND → cade
    // =====================================================
    if (me == P_SAND) {
        int down = idx + w;

        if (y < h-1 && in[down] == P_EMPTY) {
            out[idx] = P_EMPTY;
            out[down] = P_SAND;
            return;
        }

        int down_l = down - 1;
        int down_r = down + 1;

        if (left_first) {
            if (x > 0 && y < h-1 && in[down_l] == P_EMPTY) {
                out[idx] = P_EMPTY;
                out[down_l] = P_SAND;
                return;
            }
            if (x < w-1 && y < h-1 && in[down_r] == P_EMPTY) {
                out[idx] = P_EMPTY;
                out[down_r] = P_SAND;
                return;
            }
        } else {
            if (x < w-1 && y < h-1 && in[down_r] == P_EMPTY) {
                out[idx] = P_EMPTY;
                out[down_r] = P_SAND;
                return;
            }
            if (x > 0 && y < h-1 && in[down_l] == P_EMPTY) {
                out[idx] = P_EMPTY;
                out[down_l] = P_SAND;
                return;
            }
        }
    }

    // =====================================================
    // CASE 3: WATER → cade + laterale
    // =====================================================
    if (me == P_WATER) {
        int down = idx + w;

        if (y < h-1 && in[down] == P_EMPTY) {
            out[idx] = P_EMPTY;
            out[down] = P_WATER;
            return;
        }

        int down_l = down - 1;
        int down_r = down + 1;

        if (left_first) {
            if (x > 0 && y < h-1 && in[down_l] == P_EMPTY) {
                out[idx] = P_EMPTY;
                out[down_l] = P_WATER;
                return;
            }
            if (x < w-1 && y < h-1 && in[down_r] == P_EMPTY) {
                out[idx] = P_EMPTY;
                out[down_r] = P_WATER;
                return;
            }
        } else {
            if (x < w-1 && y < h-1 && in[down_r] == P_EMPTY) {
                out[idx] = P_EMPTY;
                out[down_r] = P_WATER;
                return;
            }
            if (x > 0 && y < h-1 && in[down_l] == P_EMPTY) {
                out[idx] = P_EMPTY;
                out[down_l] = P_WATER;
                return;
            }
        }

        // movimento laterale
        int left  = idx - 1;
        int right = idx + 1;

        if (left_first) {
            if (x > 0 && in[left] == P_EMPTY) {
                out[idx] = P_EMPTY;
                out[left] = P_WATER;
                return;
            }
            if (x < w-1 && in[right] == P_EMPTY) {
                out[idx] = P_EMPTY;
                out[right] = P_WATER;
                return;
            }
        } else {
            if (x < w-1 && in[right] == P_EMPTY) {
                out[idx] = P_EMPTY;
                out[right] = P_WATER;
                return;
            }
            if (x > 0 && in[left] == P_EMPTY) {
                out[idx] = P_EMPTY;
                out[left] = P_WATER;
                return;
            }
        }
    }
}*/
__global__ void sand_step_kernel(
    const unsigned char* in,
    unsigned char* out,
    int w, int h,
    int gen,
    unsigned char* clock)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;
    if (!active_cell(x, y, gen)) return;

    int idx = y * w + x;

    if (clock[idx]) return;  // già aggiornato

    unsigned char me = in[idx];
    out[idx] = me;
    mark_updated(clock, idx);

    bool left_first = (gen % 2 == 0);

    // ===================== SAND =====================
    if (me == P_SAND) {
        int below = idx + w;
        if (y < h-1) {
            unsigned char cell_below = wrap_get(in, out, clock, below);

            if (cell_below == P_EMPTY) {
                out[below] = P_SAND;
                out[idx] = P_EMPTY;
                mark_updated(clock, below);
                return;
            }

            int dir = left_first ? -1 : 1;
            int d1 = below + dir;
            int d2 = below - dir;

            if (x + dir >= 0 && x + dir < w &&
                wrap_get(in, out, clock, d1) == P_EMPTY) {
                out[d1] = P_SAND;
                out[idx] = P_EMPTY;
                mark_updated(clock, d1);
                return;
            }

            if (x - dir >= 0 && x - dir < w &&
                wrap_get(in, out, clock, d2) == P_EMPTY) {
                out[d2] = P_SAND;
                out[idx] = P_EMPTY;
                mark_updated(clock, d2);
                return;
            }

            if (cell_below == P_WATER) {
                if ((x + y + gen) % 2 == 0) {
                    out[below] = P_SAND;
                    out[idx] = P_WATER;
                    mark_updated(clock, below);
                }
                return;
            }
        }
    }

    // ===================== WATER =====================
    if (me == P_WATER) {
        int below = idx + w;

        if (y < h-1) {
            unsigned char cell_below = wrap_get(in, out, clock, below);

            if (cell_below == P_EMPTY) {
                out[below] = P_WATER;
                out[idx] = P_EMPTY;
                mark_updated(clock, below);
                return;
            }

            int left = idx - 1;
            int right = idx + 1;
            int dleft = below - 1;
            int dright = below + 1;

            int first_d = left_first ? dleft : dright;
            int second_d = left_first ? dright : dleft;

            if (x > 0 && wrap_get(in, out, clock, first_d) == P_EMPTY) {
                out[first_d] = P_WATER;
                out[idx] = P_EMPTY;
                mark_updated(clock, first_d);
                return;
            }

            if (x < w-1 && wrap_get(in, out, clock, second_d) == P_EMPTY) {
                out[second_d] = P_WATER;
                out[idx] = P_EMPTY;
                mark_updated(clock, second_d);
                return;
            }

            int h1 = left_first ? right : left;
            int h2 = left_first ? left : right;

            if ((left_first ? x < w-1 : x > 0) &&
                wrap_get(in, out, clock, h1) == P_EMPTY) {
                out[h1] = P_WATER;
                out[idx] = P_EMPTY;
                mark_updated(clock, h1);
                return;
            }

            if ((left_first ? x > 0 : x < w-1) &&
                wrap_get(in, out, clock, h2) == P_EMPTY) {
                out[h2] = P_WATER;
                out[idx] = P_EMPTY;
                mark_updated(clock, h2);
                return;
            }
        }
    }
}
