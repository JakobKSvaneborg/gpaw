#include "magma_python_interface.h"
#include "../pyarray_utils.hpp"
#include "magma_gpaw.hpp"

#include <cassert>

/* Converts uplo string 'U'/'L' to appropriate Magma type.
This also takes care of convention difference between Numpy/Cupy arrays (C-style, row major)
and Magma arrays (Fortran/LAPACK style, column major).
So 'L' in Python conventions actually means MagmaUpper. */
static magma_uplo_t get_magma_uplo(char* in_uplo_str)
{
    assert((strcmp(in_uplo_str, "L") == 0 || strcmp(in_uplo_str, "U") == 0)
        && "Invalid UPLO");

    return strcmp(in_uplo_str, "L") == 0 ? MagmaUpper : MagmaLower;
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

extern "C"
PyObject* eigh_magma_cpu(PyObject* self, PyObject* args)
{
    PyObject *in_matrix_obj;
    char* in_uplo;
    if (!PyArg_ParseTuple(args, "Os", &in_matrix_obj, &in_uplo))
    {
        return NULL;
    }

    if (!PyArray_Check(in_matrix_obj))
    {
        PyErr_SetString(PyExc_TypeError, "Input must be a numpy array");
        return NULL;
    }

    PyArrayObject* in_matrix = reinterpret_cast<PyArrayObject*>(in_matrix_obj);
    assert(in_matrix && gpaw::Array_NDIM(in_matrix) == 2);
    assert(gpaw::Array_DIM(in_matrix, 0) == gpaw::Array_DIM(in_matrix, 1));

    const int matrix_numpy_dtype = PyArray_TYPE(in_matrix);

    MagmaPythonContext python_context = decide_solver_type(matrix_numpy_dtype);
    if (python_context.solver_type == EighSolverType::eNone)
    {
        printf("Unknown numpy dtype. ID was: %d\n", matrix_numpy_dtype);
        PyErr_SetString(PyExc_TypeError, "Invalid or unsupported dtype to eigh_magma()");
        return NULL;
    }

    // Figure out input array dimensions and make output arrays of correct shape

    const npy_intp size_np = PyArray_DIM(in_matrix, 0);

    if (size_np <= 0)
    {
        PyErr_WarnEx(PyExc_RuntimeWarning,
            "Empty array?",
            1
        );
        return NULL;
    }

    const npy_intp* dims = PyArray_DIMS(in_matrix);

    PyObject* eigvals = PyArray_SimpleNew(1, &size_np, python_context.numpy_eigval_dtype);
    PyObject* eigvecs = PyArray_SimpleNew(2, dims, python_context.numpy_eigvec_dtype);

    assert(eigvals);
    assert(eigvecs);

    MagmaEighContext solver_context;
    solver_context.solver_type = python_context.solver_type;
    solver_context.matrix_size = static_cast<magma_int_t>(size_np);
    solver_context.matrix_lda = solver_context.matrix_size;
    solver_context.uplo = get_magma_uplo(in_uplo);
    solver_context.jobz = MagmaVec; // Always do eigenvectors

    const EighErrorType status = magma_eigh_cpu(
        solver_context,
        PyArray_DATA(in_matrix),
        PyArray_DATA((PyArrayObject*)eigvals),
        PyArray_DATA((PyArrayObject*)eigvecs)
    );

    assert(status != EighErrorType::eInvalidArgument && "Invalid input to MAGMA solver");
    if (status == EighErrorType::eFailedToConverge)
    {
        PyErr_WarnEx(PyExc_RuntimeWarning,
            "MAGMA eigensolver failed to converge",
            1
        );
    }

    PyObject* result = PyTuple_Pack(2, eigvals, eigvecs);

    Py_DECREF(eigvals);
    Py_DECREF(eigvecs);

    return result;
}

// CUPY doesn't provide a nice C-interface to array creation like Numpy, so need to do tricks.
// We require that the user allocates and passes valid CUPY arrays from the
// Python side for both inputs AND outputs. We parse them here and pass the
// underlying memory pointers to an internal function that does the work, ie.
// calls MAGMA. Output is written to the buffers that were passed from Python.

extern "C"
PyObject* eigh_magma_gpu(PyObject* self, PyObject* args)
{
    PyObject* in_matrix_cupy;
    char* in_uplo;

    // Must be allocated on Python side. Asserts below verify that the dtypes and sizes are OK
    PyObject* inout_eigvals_cupy;
    PyObject* inout_eigvecs_cupy;

    if (!PyArg_ParseTuple(args, "OsOO", &in_matrix_cupy, &in_uplo, &inout_eigvals_cupy,
        &inout_eigvecs_cupy))
    {
        return NULL;
    }

    assert(gpaw::Array_NDIM(in_matrix_cupy) == 2);
    assert(gpaw::Array_DIM(in_matrix_cupy, 0) == gpaw::Array_DIM(in_matrix_cupy, 1));

    // Matrix size N
    const int64_t n = gpaw::Array_DIM(in_matrix_cupy, 0);
    const int matrix_numpy_dtype = gpaw::Array_TYPE(in_matrix_cupy);

    assert(gpaw::Array_NDIM(inout_eigvecs_cupy) == 2);
    assert(gpaw::Array_DIM(inout_eigvecs_cupy, 0) == n && gpaw::Array_DIM(inout_eigvecs_cupy, 1) == n);

    assert(gpaw::Array_NDIM(inout_eigvals_cupy) == 1);
    assert(gpaw::Array_DIM(inout_eigvals_cupy, 0) == n);

    MagmaPythonContext python_context = decide_solver_type(matrix_numpy_dtype);
    if (python_context.solver_type == EighSolverType::eNone)
    {
        printf("Unknown numpy dtype. ID was: %d\n", matrix_numpy_dtype);
        PyErr_SetString(PyExc_TypeError, "Invalid or unsupported dtype to eigh_magma()");
        return NULL;
    }

    assert(gpaw::Array_TYPE(inout_eigvals_cupy) == python_context.numpy_eigval_dtype);
    assert(gpaw::Array_TYPE(inout_eigvecs_cupy) == python_context.numpy_eigvec_dtype);


    MagmaEighContext solver_context;
    solver_context.solver_type = python_context.solver_type;
    solver_context.matrix_size = static_cast<magma_int_t>(n);
    solver_context.matrix_lda = solver_context.matrix_size;
    solver_context.uplo = get_magma_uplo(in_uplo);
    solver_context.jobz = MagmaVec; // Always do eigenvectors

    const EighErrorType status = magma_eigh_gpu(
        solver_context,
        gpaw::Array_DATA<void>(in_matrix_cupy),
        gpaw::Array_DATA<void>(inout_eigvals_cupy),
        gpaw::Array_DATA<void>(inout_eigvecs_cupy)
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
