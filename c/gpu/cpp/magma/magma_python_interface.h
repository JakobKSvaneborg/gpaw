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

/* Solves symmetric/Hermitian eigenvalue problem on the CPU.
Takes 1 Numpy array as input (in_matrix) and outputs eigvals, eigvecs tuple.
*/
CLINKAGE PyObject* eigh_magma_cpu(PyObject* self, PyObject* args);

/* Solves symmetric/Hermitian eigenvalue problem on the GPU.
Takes 3 CuPy arrays as input (in_matrix, out_eigvals, out_eigvecs)
and assumes them to already be the correct size.
*/
CLINKAGE PyObject* eigh_magma_gpu(PyObject* self, PyObject* args);

#endif
