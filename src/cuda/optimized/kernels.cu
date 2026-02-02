#include "kernels.cuh"
// Checkerboard helper
__device__ inline bool active_cell(int x, int y, int gen) {
    return ((x + y + gen) & 1) == 0;
}

__global__ void sand_step_kernel(const unsigned char* in, unsigned char* out, int w, int h, int gen){

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;
    if (!active_cell(x, y, gen)) return;

    int idx = y * w + x;
    unsigned char me = in[idx];

    // Default: copy
    out[idx] = me;

    // Non può cambiare se è muro
    if (me == P_WALL) return;

    // ---- CASO 1: SONO VUOTO → posso ricevere sabbia ----
    if (me == P_EMPTY) {
        int up    = idx - w;
        int up_l  = up - 1;
        int up_r  = up + 1;

        // sopra
        if (y > 0 && in[up] == P_SAND) {
            out[idx] = P_SAND;
            out[up]  = P_EMPTY;
            return;
        }

        // diagonali (alternanza come nel logic.c)
        bool left_first = (gen & 1);

        if (left_first) {
            if (x > 0 && y > 0 && in[up_l] == P_SAND) {
                out[idx] = P_SAND;
                out[up_l] = P_EMPTY;
                return;
            }
            if (x < w-1 && y > 0 && in[up_r] == P_SAND) {
                out[idx] = P_SAND;
                out[up_r] = P_EMPTY;
                return;
            }
        } else {
            if (x < w-1 && y > 0 && in[up_r] == P_SAND) {
                out[idx] = P_SAND;
                out[up_r] = P_EMPTY;
                return;
            }
            if (x > 0 && y > 0 && in[up_l] == P_SAND) {
                out[idx] = P_SAND;
                out[up_l] = P_EMPTY;
                return;
            }
        }
    }

    // ---- CASO 2: SONO SABBIA → posso cadere ----
    if (me == P_SAND) {
        int down = idx + w;

        if (y < h-1 && in[down] == P_EMPTY) {
            out[idx] = P_EMPTY;
            out[down] = P_SAND;
            return;
        }

        bool left_first = (gen & 1);

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
}