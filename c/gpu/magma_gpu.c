#if defined(GPAW_WITH_MAGMA) && defined(GPAW_GPU)

#include "../extensions.h"

// Define magic to enable custom Array_** macros for CUPY arrays
#define GPAW_ARRAY_DISABLE_NUMPY
#define GPAW_ARRAY_ALLOW_CUPY
#include "../array.h"
#undef GPAW_ARRAY_DISABLE_NUMPY

#include "gpu.h"
#include "gpu-complex.h"
#include "../magma_gpaw.h"

#include <assert.h>
#include <string.h>

// CUPY doesn't provide a nice C-interface like Numpy, so need to do tricks.
// We require that the user allocates and passes valid CUPY arrays from the
// Python side for both inputs AND outputs. We parse them here and pass the
// underlying memory pointers to an internal function that does the work, ie.
// calls MAGMA. Output is written to the buffers that were passed from Python.

static magma_int_t _eigh_magma_dsyevd_gpu(int matrix_size, magma_uplo_t uplo,
    double* in_matrix, double* inout_eigvals, double* inout_eigvects)
{
    // Caller is responsible for ensuring that buffers are already correct size

    // Input matrix to MAGMA gets overriden by eigenvectors, so take a copy here.
    gpuMemcpy(inout_eigvects, in_matrix, matrix_size * matrix_size * sizeof(double), gpuMemcpyDeviceToDevice);

    const magma_vec_t jobz = MagmaVec; // always compute eigenvectors
    const magma_int_t lda = matrix_size;

    syevd_workgroup workgroup = {};
    // Query
    double work_temp;
    magma_int_t iwork_temp;
    magma_int_t status;
    magma_dsyevd_gpu(jobz, uplo, matrix_size, NULL, lda, NULL,
        NULL, lda, &work_temp, -1, &iwork_temp, -1, &status
    );

    assert(status == 0 && "magma_dsyevd_gpu query failed");
    workgroup.lwork = (magma_int_t) work_temp;
    workgroup.liwork = iwork_temp;

    assert(workgroup.lwork > 0);
    assert(workgroup.liwork > 0);

    // All buffers apart from the input matrix are in HOST memory
    workgroup.work = malloc(workgroup.lwork * sizeof(double));
    workgroup.iwork = malloc(workgroup.liwork * sizeof(magma_int_t));

    double* h_wA = malloc(matrix_size * lda * sizeof(double));
    double* h_eigvals = malloc(matrix_size * sizeof(double));

    magma_dsyevd_gpu(jobz, uplo, matrix_size, inout_eigvects, lda,
        h_eigvals, h_wA, lda, workgroup.work, workgroup.lwork,
        workgroup.iwork, workgroup.liwork, &status
    );

    // copy eigenvalues to device output buffer
    gpuMemcpy(inout_eigvals, h_eigvals, matrix_size * sizeof(double), gpuMemcpyHostToDevice);


    free(h_wA);
    free(workgroup.work);
    free(workgroup.iwork);
    free(h_eigvals);

    return status;
}

/* GPU version, operates on CUPY arrays.
* Assumes that the input is a valid CUPY array, allocated in GPU memory.
* Validations should be done on Python side before calling this.
*/
PyObject* eigh_magma_dsyevd_gpu(PyObject* self, PyObject* args)
{
    PyObject *in_matrix;
    char* in_uplo;

    // Must be allocated on python side
    PyObject *inout_eigvals;
    PyObject *inout_eigvects;

    if (!PyArg_ParseTuple(args, "OsOO", &in_matrix, &in_uplo, &inout_eigvals,
        &inout_eigvects))
    {
        return NULL;
    }

    assert(Array_NDIM(in_matrix) == 2);
    assert(Array_DIM(in_matrix, 0) == Array_DIM(in_matrix, 1));
    assert(Array_ITEMSIZE(in_matrix) == sizeof(double));

    assert(Array_NDIM(inout_eigvects) == 2);
    assert(Array_DIM(inout_eigvects, 0) == Array_DIM(inout_eigvects, 1));
    assert(Array_ITEMSIZE(inout_eigvects) == sizeof(double));

    assert(Array_NDIM(inout_eigvals) == 1);
    assert(Array_ITEMSIZE(inout_eigvals) == sizeof(double));

    const size_t n = Array_DIM(in_matrix, 0);
    const magma_uplo_t uplo = get_magma_uplo(in_uplo);

    magma_int_t status = _eigh_magma_dsyevd_gpu(
        n,
        uplo,
        Array_DATA(in_matrix),
        Array_DATA(inout_eigvals),
        Array_DATA(inout_eigvects)
    );

    assert(status >= 0 && "Invalid input to MAGMA solver");
    if (status > 0)
    {
        PyErr_WarnEx(PyExc_RuntimeWarning,
            "MAGMA eigensolver failed to converge",
            1
        );
    }

    Py_RETURN_NONE;
}


#endif
