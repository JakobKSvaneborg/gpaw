#ifndef GPU_RUNTIME_H
#define GPU_RUNTIME_H

#include "gpaw_utils.h"

#ifdef GPAW_CUDA
#include "cuda.h"
#endif
#ifdef GPAW_HIP
#include "hip.h"
#endif

#include <stdio.h>

#define gpuSafeCall(err)          __gpuSafeCall(err, __FILE__, __LINE__)
#define gpublasSafeCall(err)      __gpublasSafeCall(err, __FILE__, __LINE__)

static inline int __gpuSafeCall(gpuError_t err,
    const char *file, int line)
{
    if (gpuSuccess != err) {
        char str[100];
        snprintf(str, 100, "%s(%i): GPU error: %s.\n",
        file, line, gpuGetErrorString(err));
        gpaw_set_runtime_error(str);
        fprintf(stderr, "%s", str);
    }
    return err;
}

static inline gpublasStatus_t __gpublasSafeCall(gpublasStatus_t err,
                    const char *file, int line)
{
    if (GPUBLAS_STATUS_SUCCESS != err) {
        char str[100];
        switch (err) {
        case GPUBLAS_STATUS_NOT_INITIALIZED:
        snprintf(str, 100,
        "%s(%i): GPU BLAS error: NOT INITIALIZED.\n",
        file, line);
        break;
        case GPUBLAS_STATUS_ALLOC_FAILED:
        snprintf(str, 100,
        "%s(%i): GPU BLAS error: ALLOC FAILED.\n",
        file, line);
        break;
        case GPUBLAS_STATUS_INVALID_VALUE:
        snprintf(str, 100,
        "%s(%i): GPU BLAS error: INVALID VALUE.\n",
        file, line);
        break;
        case GPUBLAS_STATUS_ARCH_MISMATCH:
        snprintf(str, 100,
        "%s(%i): GPU BLAS error: ARCH MISMATCH.\n",
        file, line);
        break;
        case GPUBLAS_STATUS_MAPPING_ERROR:
        snprintf(str, 100,
        "%s(%i): GPU BLAS error: MAPPING ERROR.\n",
        file, line);
        break;
        case GPUBLAS_STATUS_EXECUTION_FAILED:
        snprintf(str, 100,
        "%s(%i): GPU BLAS error: EXECUTION FAILED.\n",
        file, line);
        break;
        case GPUBLAS_STATUS_INTERNAL_ERROR:
        snprintf(str, 100,
        "%s(%i): GPU BLAS error: INTERNAL ERROR.\n",
        file, line);
        break;
        default:
        snprintf(str, 100,
        "%s(%i): GPU BLAS error: UNKNOWN ERROR '%X'.\n",
        file, line, err);
        }
        gpaw_set_runtime_error(str);
        fprintf(stderr, "%s", str);
    }
    return err;
}

#endif
