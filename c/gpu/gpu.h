#ifndef GPU_GPU_H
#define GPU_GPU_H

#include <stdio.h>
#include <float.h>
#include "gpu-runtime.h"

// Numpy datatype defines
#define NP_FLOAT                  (11)
#define NP_DOUBLE                 (12)
#define NP_FLOAT_COMPLEX          (14)
#define NP_DOUBLE_COMPLEX         (15)

#define GPU_BLOCKS_MIN            (16)
#define GPU_BLOCKS_MAX            (96)
#define GPU_DEFAULT_BLOCK_X       (32)
#define GPU_DEFAULT_BLOCK_Y       (16)

#define GPU_ASYNC_SIZE            (8*1024)
#define GPU_RJOIN_SIZE            (16*1024)
#define GPU_SJOIN_SIZE            (16*1024)
#define GPU_RJOIN_SAME_SIZE       (96*1024)
#define GPU_SJOIN_SAME_SIZE       (96*1024)
#define GPU_OVERLAP_SIZE          (GPU_ASYNC_SIZE)

#define GPU_ERROR_ABS_TOL         (1e-13)
#define GPU_ERROR_ABS_TOL_EXCT    (DBL_EPSILON)

#define GPAW_BOUNDARY_NORMAL      (1<<(0))
#define GPAW_BOUNDARY_SKIP        (1<<(1))
#define GPAW_BOUNDARY_ONLY        (1<<(2))
#define GPAW_BOUNDARY_X0          (1<<(3))
#define GPAW_BOUNDARY_X1          (1<<(4))
#define GPAW_BOUNDARY_Y0          (1<<(5))
#define GPAW_BOUNDARY_Y1          (1<<(6))
#define GPAW_BOUNDARY_Z0          (1<<(7))
#define GPAW_BOUNDARY_Z1          (1<<(8))

#define GPU_PITCH                 (16)  /* in doubles */
#define NEXTPITCHDIV(n) \
        (((n) > 0) ? ((n) + GPU_PITCH - 1 - ((n) - 1) % GPU_PITCH) : 0)

#ifndef MAX
#  define MAX(a,b)  (((a) > (b)) ? (a) : (b))
#endif
#ifndef MIN
#  define MIN(a,b)  (((a) < (b)) ? (a) : (b))
#endif

typedef struct
{
    int ncoefs;
    double* coefs_gpu;
    long* offsets_gpu;
    int ncoefs0;
    double* coefs0_gpu;
    int ncoefs1;
    double* coefs1_gpu;
    int ncoefs2;
    double* coefs2_gpu;
    double coef_relax;
    long n[3];
    long j[3];
} bmgsstencil_gpu;

#ifndef BMGS_H
typedef struct
{
    int ncoefs;
    double* coefs;
    long* offsets;
    long n[3];
    long j[3];
} bmgsstencil;
#endif

static inline unsigned int nextPow2(unsigned int x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

#define BLOCK_GRID(hc_size)                                           \
    int blockx = MIN((int)nextPow2(hc_size.z),                        \
                     BLOCK_MAX);                                      \
    int blocky = MIN(MIN((int)nextPow2(hc_size.y),                    \
                         BLOCK_TOTALMAX / blockx),                    \
                     BLOCK_MAX);                                      \
    dim3 dimBlock(blockx, blocky);                                    \
    int gridx = ((hc_size.z + dimBlock.x - 1) / dimBlock.x);          \
    int xdiv = MAX(1, MIN(hc_size.x, GRID_MAX / gridx));              \
    gridx = xdiv * gridx;                                             \
    int gridy = blocks * ((hc_size.y + dimBlock.y - 1) / dimBlock.y); \
    dim3 dimGrid(gridx, gridy);                                       \


#endif
