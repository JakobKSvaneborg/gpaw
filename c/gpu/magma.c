#ifdef GPAW_WITH_MAGMA
#include "magma.h"
#include <magma_v2.h>
#include <magma_auxiliary.h>

#include "../extensions.h"
#include "gpu.h"
#include "gpu-complex.h"

#include <assert.h>
#include <string.h>

static magma_queue_t queue;

struct MagmaDiagonalizationInfo
{
    magma_vec_t jobz;
    magma_uplo_t uplo;
    // N x N matrix
    magma_int_t n;
    magma_int_t lda;

    magma_int_t lwork;
    magma_int_t liwork;

};

// Find optimal workgroup sizes
static magma_int_t query_magma_work_sizes_syevd(
    MagmaDiagonalizationInfo *in_out_info)
{
    double work_temp;
    magma_int_t iwork_temp;

    magma_int_t status;

    magma_dsyevd_gpu(in_out_info.vec, in_out_info.uplo, in_out_info.n, NULL,
        in_out_info.lda, NULL, NULL, lda, &work_temp, -1, &iwork_temp, -1,
        &status);

    in_out_info.lwork = (magma_int_t) work_temp;
    in_out_info.liwork = iwork_temp;

    return status;
}


PyObject* eigh_syevd_magma(PyObject* self, PyObject* args)
{
    PyObject *in_mat_gpu;
    char* uplo;

    if (!PyArg_ParseTuple(args, "Os", &mat_gpu, &uplo))
    {
        return NULL;
    }

    // Do more informative validations on Python side, here just debug assert

    // Should be double-precision, real square matrix
    assert(Array_ITEMSIZE(in_mat_gpu) == sizeof(double));
    assert(!Array_ISCOMPLEX(in_mat_gpu));
    assert(Array_NDIM(in_mat_gpu) == 2);
    assert(Array_DIM(in_mat_gpu, 0) == Array_DIM(in_mat_gpu, 1));

    assert(strcmp(uplo, "L") == 0 || strcmp(uplo, "U") == 0);

    // gets overriden by MAGMA, do we want this?
    double* mat_gpu = Array_DATA(in_mat_gpu);

    if (PyErr_Occurred())
    {
        return NULL;
    }

    MagmaDiagonalizationInfo info;
    info.jobz = MagmaVec;
    info.uplo = MagmaLower ? strcmp(uplo, "L") == 0 : MagmaUpper;
    info.n = (magma_int_t) Array_DIM(in_mat_gpu, 0);
    info.lda = n;

    // Query work sizes. TODO should we cache?
    magma_int_t status = query_magma_work_sizes_syevd(&info);
    assert(status == 0 && "MAGMA work size query failed");

    printf("MAGMA: status %d lwork %d liwork %d\n", status, info.lwork, info.liwork);

    /*
    // this assumes single GPU
    magma_dsyevd_gpu(magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n, matrix_dtype *dA,
        magma_int_t ldda, real_t *w, matrix_dtype *wA, magma_int_t ldwa, matrix_dtype *work, magma_int_t lwork,
        magma_int_t *iwork, magma_int_t liwork, magma_int_t *info);
    */

    // temp
    Py_RETURN_NONE;
}

#endif
