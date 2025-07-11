#include "magma_python_interface.h"
#include "../pyarray_utils.hpp"
#include "magma_gpaw.hpp"

#include <cassert>

static magma_uplo_t get_magma_uplo(char* in_uplo_str)
{
    assert((strcmp(in_uplo_str, "L") == 0 || strcmp(in_uplo_str, "U") == 0)
        && "Invalid UPLO");

    return strcmp(in_uplo_str, "L") == 0 ? MagmaLower : MagmaUpper;
}

struct MagmaPythonContext
{
    int numpy_eigval_dtype;
    int numpy_eigvec_dtype;
    EighSolverType solver_type;
};

// Figure out solver type and eigenvector/eigenvalue Numpy dtypes based on dtype of the matrix to be diagonalized
static MagmaPythonContext decide_solver_type(int matrix_numpy_dtype)
{
    MagmaPythonContext context;
    switch (matrix_numpy_dtype)
    {
    case NPY_FLOAT:
        context.solver_type = EighSolverType::eSsyevd;
        context.numpy_eigval_dtype = NPY_FLOAT;
        context.numpy_eigvec_dtype = NPY_FLOAT;
        break;

    case NPY_DOUBLE:
        context.solver_type = EighSolverType::eDsyevd;
        context.numpy_eigval_dtype = NPY_DOUBLE;
        context.numpy_eigvec_dtype = NPY_DOUBLE;
        break;

    case NPY_CFLOAT:
        context.solver_type = EighSolverType::eCheevd;
        context.numpy_eigval_dtype = NPY_FLOAT;
        context.numpy_eigvec_dtype = NPY_CFLOAT;
        break;

    case NPY_CDOUBLE:
        context.solver_type = EighSolverType::eZheevd;
        context.numpy_eigval_dtype = NPY_DOUBLE;
        context.numpy_eigvec_dtype = NPY_CDOUBLE;
        break;

    default:
        // Invalid dtype
        context.solver_type = EighSolverType::eNone;
        break;
    }

    return context;
}

CLINKAGE PyObject* eigh_magma_numpy(PyObject* self, PyObject* args)
{
    PyObject *inout_matrix;
    PyObject* inout_eigvals;
    char* in_uplo;
    int num_gpus;
    if (!PyArg_ParseTuple(args, "OOs", &inout_matrix, &inout_eigvals, &in_uplo, &num_gpus))
    {
        return NULL;
    }

    if (!PyArray_Check(inout_matrix) || !PyArray_Check(inout_eigvals))
    {
        PyErr_SetString(PyExc_TypeError, "Inputs must be numpy arrays");
        return NULL;
    }

    PyArrayObject* inout_matrix_numpy = reinterpret_cast<PyArrayObject*>(inout_matrix);
    assert(inout_matrix_numpy && gpaw::Array_NDIM(inout_matrix_numpy) == 2);
    assert(gpaw::Array_DIM(inout_matrix_numpy, 0) == gpaw::Array_DIM(inout_matrix_numpy, 1));

    const int matrix_numpy_dtype = PyArray_TYPE(inout_matrix_numpy);
    const int64_t matrix_size = gpaw::Array_DIM(inout_matrix_numpy, 0);

    PyArrayObject* inout_eigvals_numpy = reinterpret_cast<PyArrayObject*>(inout_eigvals);

    assert(gpaw::Array_NDIM(inout_eigvals_numpy) == 1);
    assert(gpaw::Array_DIM(inout_eigvals_numpy, 0) == matrix_size);

    MagmaPythonContext python_context = decide_solver_type(matrix_numpy_dtype);
    if (python_context.solver_type == EighSolverType::eNone)
    {
        printf("Unknown numpy dtype. ID was: %d\n", matrix_numpy_dtype);
        PyErr_SetString(PyExc_TypeError, "Invalid or unsupported dtype to eigh_magma()");
        return NULL;
    }

    MagmaEighContext solver_context;
    solver_context.solver_type = python_context.solver_type;
    solver_context.matrix_size = static_cast<magma_int_t>(matrix_size);
    solver_context.matrix_lda = solver_context.matrix_size;
    solver_context.uplo = get_magma_uplo(in_uplo);
    solver_context.jobz = MagmaVec; // Always do eigenvectors

    assert(num_gpus > 0);
    // TODO validate this
    solver_context.num_gpus = num_gpus;

    const EighErrorType status = magma_eigh_host(
        solver_context,
        PyArray_DATA(inout_matrix_numpy),
        PyArray_DATA(inout_eigvals_numpy)
    );

    assert(status != EighErrorType::eInvalidArgument && "Invalid input to MAGMA solver");
    if (status == EighErrorType::eFailedToConverge)
    {
        PyErr_WarnEx(PyExc_RuntimeWarning,
            "MAGMA eigensolver failed to converge",
            1
        );
    }

    Py_RETURN_NONE;
}

// CUPY doesn't provide a nice C-interface to array creation like Numpy, so need to do tricks.
// We require that the user allocates and passes valid CUPY arrays from the
// Python side for both inputs AND outputs. We parse them here and pass the
// underlying memory pointers to an internal function that does the work, ie.
// calls MAGMA. Output is written to the buffers that were passed from Python.

CLINKAGE PyObject* eigh_magma_cupy(PyObject* self, PyObject* args)
{
    // Must be allocated on Python side. Asserts below verify that the dtypes and sizes are OK
    PyObject* inout_matrix_cupy;
    PyObject* inout_eigvals_cupy;
    char* in_uplo;

    if (!PyArg_ParseTuple(args, "OOs", &inout_matrix_cupy, &inout_eigvals_cupy, &in_uplo))
    {
        return NULL;
    }

    assert(gpaw::Array_NDIM(inout_matrix_cupy) == 2);
    assert(gpaw::Array_DIM(inout_matrix_cupy, 0) == gpaw::Array_DIM(inout_matrix_cupy, 1));

    const int64_t matrix_size = gpaw::Array_DIM(inout_matrix_cupy, 0);
    const int matrix_numpy_dtype = gpaw::Array_TYPE(inout_matrix_cupy);

    assert(gpaw::Array_NDIM(inout_eigvals_cupy) == 1);
    assert(gpaw::Array_DIM(inout_eigvals_cupy, 0) == matrix_size);

    MagmaPythonContext python_context = decide_solver_type(matrix_numpy_dtype);
    if (python_context.solver_type == EighSolverType::eNone)
    {
        printf("Unknown numpy dtype. ID was: %d\n", matrix_numpy_dtype);
        PyErr_SetString(PyExc_TypeError, "Invalid or unsupported dtype to eigh_magma()");
        return NULL;
    }

    assert(gpaw::Array_TYPE(inout_eigvals_cupy) == python_context.numpy_eigval_dtype);

    MagmaEighContext solver_context;
    solver_context.solver_type = python_context.solver_type;
    solver_context.matrix_size = static_cast<magma_int_t>(matrix_size);
    solver_context.matrix_lda = solver_context.matrix_size;
    solver_context.uplo = get_magma_uplo(in_uplo);
    solver_context.jobz = MagmaVec; // Always do eigenvectors
    // This only supports single GPU
    solver_context.num_gpus = 1;

    const EighErrorType status = magma_eigh_gpu(
        solver_context,
        gpaw::Array_DATA<void*>(inout_matrix_cupy),
        gpaw::Array_DATA<void*>(inout_eigvals_cupy)
    );

    assert(status != EighErrorType::eInvalidArgument && "Invalid input to MAGMA solver");
    if (status == EighErrorType::eFailedToConverge)
    {
        PyErr_WarnEx(PyExc_RuntimeWarning,
            "MAGMA eigensolver failed to converge",
            1
        );
    }

    Py_RETURN_NONE;
}
