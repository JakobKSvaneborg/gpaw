#ifndef MAGMA_PYTHON_INTERFACE_H
#define MAGMA_PYTHON_INTERFACE_H

// C99 compliant header that can safely be included from main GPAW.

#ifdef __cplusplus
    #define CLINKAGE extern "C"
#else
    #define CLINKAGE
#endif

// MAGMA needs stdbool.h but it is not properly included by their own headers.
// Can remove this include once it's fixed in MAGMA.
// See https://github.com/icl-utk-edu/magma/pull/41
#include <stdbool.h>
#include <magma_v2.h>
#include <Python.h>

/* Initializes MAGMA library. Must be called come AFTER any calls to cudaSetValidDevices
* and cudaSetDeviceFlags. Call only if GPUs are available.
*/
CLINKAGE void gpaw_magma_init();
CLINKAGE void gpaw_magma_finalize();

/* Solves symmetric/Hermitian eigenvalue on GPU, using Numpy arrays as input
and output. In other words, all input and output arrays are in CPU memory and
contiguous. Multiple GPUs can be used if requested, in which case MAGMA handles
the distribution to GPUs internally. Note that multi-GPU use is limited to
one compute node, as MAGMA has no support for SCALAPACK-like distributed
functionality. Therefore this should be called from only one MPI process.

In practice this is wrapper around magma_zheevd() and magma_zheevd_m(),
and their SYEVD counterparts. Eigenvectors will be returned in MAGMA/SCALAPACK Fortran-like convention.

Syntax when calling from Python:
    eigh_magma_numpy(inout_matrix: np.ndarray, out_eigvals: np.ndarray, uplo: str, num_gpus: int) -> None

`inout_matrix` gets overwritten by the eigenvectors. Both arrays must be
allocated to correct size before calling this function.

Input conventions are as in Numpy, but output eigenvectors are still in Magma
conventions => take conjugate transpose on Python side to get them in Numpy convention.
*/
CLINKAGE PyObject* eigh_magma_numpy(PyObject* self, PyObject* args);

/* Solves symmetric/Hermitian eigenvalue on single GPU, using CuPy arrays as input
and output. Differences to `eigh_magma_numpy` are:
    1. Input and output are directly in GPU memory
    2. Only supports single-GPU solving

In practice this is wrapper around magma_zheevd_gpu(), and its SYEVD counterparts.
Eigenvectors will be returned in MAGMA/SCALAPACK Fortran-like convention.

Syntax when calling from Python:
    eigh_magma_cupy(inout_matrix: cp.ndarray, out_eigvals: np.ndarray, uplo: str) -> None

`inout_matrix` gets overwritten by the eigenvectors. Both arrays must be
allocated to correct size before calling this function.
Input conventions are as in Cupy, but output eigenvectors are still in Magma
conventions => take conjugate transpose on Python side to get them in Cupy convention.
*/
CLINKAGE PyObject* eigh_magma_cupy(PyObject* self, PyObject* args);

#endif
