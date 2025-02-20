#ifdef GPAW_WITH_MAGMA

#include "../extensions.h"
#include "../array.h"

// MAGMA needs stdbool.h but it is not properly included by their own headers.
// Can remove this include once it is fixed in MAGMA.
// See https://github.com/icl-utk-edu/magma/pull/41
#include <stdbool.h>

#include <magma_v2.h>
#include <magma_auxiliary.h>

#include "gpu.h"
#include "gpu-complex.h"

#include <assert.h>
#include <string.h>

/*
typedef enum {
    eNone,
    eDouble,
    eComplexDouble
} matrix_dtype;


typedef struct gpaw_magma_eigh_info
{
    matrix_dtype dtype;
    magma_vec_t jobz;
    magma_uplo_t uplo;
    // N x N matrix
    magma_int_t n;
    magma_int_t lda;
} gpaw_magma_eigh_info;
*/

PyObject* eigh_magma_dsyevd(PyObject* self, PyObject* args)
{
    PyObject *in_matrix;
    char* in_uplo;

    if (!PyArg_ParseTuple(args, "Os", &in_matrix, &in_uplo))
    {
        return NULL;
    }

    if (!PyArray_Check(in_matrix))
    {
        PyErr_SetString(PyExc_TypeError, "Input must be a numpy array");
        return NULL;
    }

    assert(Array_NDIM(in_matrix) == 2);
    assert(Array_DIM(in_matrix, 0) == Array_DIM(in_matrix, 1));

    // TEMP TEMP
    assert(Array_ITEMSIZE(in_matrix) == sizeof(double));

    const size_t n = Array_DIM(in_matrix, 0);

    PyObject *eigvals = PyArray_SimpleNew(1, (npy_intp[]){n}, NPY_DOUBLE);
    PyObject* eigvects = PyArray_SimpleNew(2, PyArray_DIMS((PyArrayObject*)in_matrix), NPY_DOUBLE);

    // TODO check alloc OK

    const magma_vec_t jobz = MagmaVec; // always compute eigenvectors
    const magma_int_t lda = n;
    const magma_uplo_t uplo = strcmp(in_uplo, "L") == 0 ? MagmaLower : MagmaUpper;

    // Copy the input matrix because MAGMA will override it with eigenvectors.
    // So we can use the eigenvector buffer both as a work copy and as output.
    double* dA = Array_DATA(eigvects);
    memcpy(dA, Array_DATA(in_matrix), n*n*sizeof(double));

    struct syevd_workgroup
    {
        magma_int_t lwork;
        magma_int_t liwork;
        double* work;
        magma_int_t* iwork;
    };

    struct syevd_workgroup workgroup = {0, 0, NULL, NULL};

    // Query optimal workgroup sizes
    double work_temp;
    magma_int_t iwork_temp;
    magma_int_t status;
    magma_dsyevd(
        jobz,
        uplo,
        n,
        NULL,
        lda,
        NULL,
        &work_temp,
        -1,
        &iwork_temp,
        -1,
        &status
    );

    assert(status == 0 && "MAGMA dsyevd query failed");
    workgroup.lwork = (magma_int_t) work_temp;
    workgroup.liwork = iwork_temp;

    assert(workgroup.lwork > 0);
    assert(workgroup.liwork > 0);

    workgroup.work = malloc(workgroup.lwork * sizeof(double));
    workgroup.iwork = malloc(workgroup.liwork * sizeof(magma_int_t));

    magma_dsyevd(
        jobz,
        uplo,
        n,
        dA,
        lda,
        Array_DATA(eigvals),
        workgroup.work,
        workgroup.lwork,
        workgroup.iwork,
        workgroup.liwork,
        &status
    );

    if (status != 0)
    {
        // ... todo
    }

    free(workgroup.work);
    free(workgroup.iwork);

    // Eigenvectors (dA) were already filled in by MAGMA

    PyObject* result = PyTuple_Pack(2, eigvals, eigvects);

    Py_DECREF(eigvals);
    Py_DECREF(eigvects);

    return result;
}

#endif
