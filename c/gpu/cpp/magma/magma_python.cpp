#include "python_utils.h"
#include "magma_python.h"
#include "magma_gpaw.hpp"
#include "magma_solvers.hpp"
#include "gpu/cpp/pyarray_utils.hpp"
#include "gpu/cpp/pybind_cupy_type_caster.hpp"
#include "gpaw_utils.h"

#include <cassert>

namespace py = pybind11;

namespace gpaw
{

struct MagmaStaticData
{
    magma_int_t num_gpus = -1;
};

static MagmaStaticData magma_static_data;

/* Converts uplo string 'U'/'L' to appropriate Magma type.
This also takes care of convention difference between Numpy/Cupy arrays (C-style, row major)
and Magma arrays (Fortran/LAPACK style, column major).
So 'L' in Python conventions actually means MagmaUpper. */
static magma_uplo_t get_magma_uplo(const std::string& uplo)
{
    if (!(uplo == "L" || uplo == "U"))
    {
        throw std::invalid_argument("Invalid UPLO, must be 'L' or 'U'");
    }

    // Handle array convention difference
    return uplo == "L" ? MagmaUpper : MagmaLower;
}

static void check_gpu_request(int64_t requested_num_gpus)
{
    if (requested_num_gpus <= 0)
    {
        throw std::invalid_argument("requested_num_gpus <= 0");
    }

    if (requested_num_gpus > (int64_t)magma_static_data.num_gpus)
    {
        char error_msg[256];
        snprintf(error_msg, sizeof(error_msg),
            "Requested %ld GPUs to a MAGMA routine but MAGMA only sees %ld.\n",
           (long)requested_num_gpus, (long)(magma_static_data.num_gpus)
        );

        throw std::invalid_argument(error_msg);
    }
}

/* Checks that the given matrix and eigenvalue arrays have matching dtypes (symmetric/Hermitian solver).
* If matrix dtype is complex<T> -> eigenvalue dtype must be T.
* If matrix dtype is T -> eigenvalue dtype must be T.
* Throws on failure. Note that integer dtypes may pass this.
*/
static void ensure_dtype_match_eigh(py::dtype matrix_dtype, py::dtype eigval_dtype)
{
    // Complex case
    if (matrix_dtype.kind() == 'c')
    {
        if (matrix_dtype.is(py::dtype::of<std::complex<double>>()))
        {
            if (!eigval_dtype.is(py::dtype::of<double>()))
            {
                throw std::invalid_argument("Matrix/eigenvalue dtype mismatch (complex double-precision)");
            }
        }
        else if (matrix_dtype.is(py::dtype::of<std::complex<float>>()))
        {
            if (!eigval_dtype.is(py::dtype::of<float>()))
            {
                throw std::invalid_argument("Matrix/eigenvalue dtype mismatch (complex single-precision)");
            }
        }
    }
    // Real case
    else if (!eigval_dtype.is(matrix_dtype))
    {
        throw std::invalid_argument("Matrix/eigenvalue mismatch (real matrix)");
    }
}

static void ensure_dtype_match_eigh(py::array matrix, py::array eigvals)
{
    ensure_dtype_match_eigh(matrix.dtype(), eigvals.dtype());
}

/* We could use the templated py::array_t<T> type here to enforce matching dtypes for the matrix and eigenvalue arrays.
* However pybind11 does implicit conversions, meaning array copies, if the input is non-conforming.
* This is bad for performance and is also not what we want, since we need to mutate the input arrays directly.
* Therefore the following uses type-erased py::array objects as inputs, which is essentially just PyArrayObject*.
*/

static void eigh_magma_numpy(
    py::array inout_matrix,
    py::array inout_eigvals,
    const py::str& uplo,
    int64_t num_gpus)
{
    check_gpu_request(num_gpus);

    if (inout_matrix.ndim() != 2 || inout_matrix.shape()[0] != inout_matrix.shape()[1])
    {
        throw std::invalid_argument("Input matrix must be an N x N matrix");
    }
    const py::ssize_t matrix_size = inout_matrix.shape()[0];

    if (inout_eigvals.ndim() != 1 || inout_eigvals.shape()[0] != matrix_size)
    {
        throw std::invalid_argument("Input array for eigenvalues must be 1D and have length N (if the matrix is N x N)");
    }

    if (!gpaw::is_c_contiguous(inout_matrix) || inout_eigvals.strides(0) != inout_eigvals.itemsize())
    {
        throw std::invalid_argument("Input matrix and eigenvalue arrays must be C-contiguous");
    }

    ensure_dtype_match_eigh(inout_matrix, inout_eigvals);

    MagmaEighContext solver_context;
    solver_context.matrix_size = static_cast<magma_int_t>(matrix_size);
    solver_context.matrix_lda = solver_context.matrix_size;
    solver_context.uplo = get_magma_uplo(uplo.cast<std::string>());
    solver_context.jobz = MagmaVec; // Always do eigenvectors

    const py::dtype dtype = inout_matrix.dtype();
    magma_int_t status;

    switch (dtype.normalized_num())
    {
        case py::dtype::num_of<float>():
            status = magma_symmetric_solver_host<float>(
                solver_context,
                static_cast<float*>(inout_matrix.mutable_data()),
                static_cast<float*>(inout_eigvals.mutable_data())
            );
            break;
        case py::dtype::num_of<double>():
            status = magma_symmetric_solver_host<double>(
                solver_context,
                static_cast<double*>(inout_matrix.mutable_data()),
                static_cast<double*>(inout_eigvals.mutable_data())
            );
            break;
        case py::dtype::num_of<std::complex<float>>():
            status = magma_hermitian_solver_host<float>(
                solver_context,
                static_cast<magmaComplex<float>*>(inout_matrix.mutable_data()),
                static_cast<float*>(inout_eigvals.mutable_data())
            );
            break;
        case py::dtype::num_of<std::complex<double>>():
            status = magma_hermitian_solver_host<double>(
                solver_context,
                static_cast<magmaComplex<double>*>(inout_matrix.mutable_data()),
                static_cast<double*>(inout_eigvals.mutable_data())
            );
            break;
    default:
        throw std::invalid_argument("Unsupported matrix dtype");
    }

    if (status < 0)
    {
        throw std::runtime_error("Invalid input to MAGMA solver at position " + std::to_string(-status));
    }
    else if (status > 0)
    {
        PyErr_WarnEx(PyExc_RuntimeWarning,
            "MAGMA eigensolver failed to converge",
            1
        );
    }
}

static void eigh_magma_cupy(
    PyDeviceArray& inout_matrix,
    PyDeviceArray& inout_eigvals,
    const py::str& uplo)
{
    if (inout_matrix.ndim() != 2 || inout_matrix.shape[0] != inout_matrix.shape[1])
    {
        throw std::invalid_argument("Input matrix must be an N x N matrix");
    }
    const py::ssize_t matrix_size = inout_matrix.shape[0];

    if (inout_eigvals.ndim() != 1 || inout_eigvals.shape[0] != matrix_size)
    {
        throw std::invalid_argument("Input array for eigenvalues must be 1D and have length N (if the matrix is N x N)");
    }

    if (!inout_matrix.is_c_contiguous() || !inout_eigvals.is_c_contiguous())
    {
        throw std::invalid_argument("Input matrix and eigenvalue arrays must be C-contiguous");
    }

    ensure_dtype_match_eigh(inout_matrix.dtype, inout_eigvals.dtype);

    MagmaEighContext solver_context;
    solver_context.matrix_size = static_cast<magma_int_t>(matrix_size);
    solver_context.matrix_lda = solver_context.matrix_size;
    solver_context.uplo = get_magma_uplo(uplo.cast<std::string>());
    solver_context.jobz = MagmaVec; // Always do eigenvectors

    const py::dtype dtype = inout_matrix.dtype;
    magma_int_t status;

    switch (dtype.normalized_num())
    {
        case py::dtype::num_of<float>():
            status = magma_symmetric_solver_gpu<float>(
                solver_context,
                static_cast<float*>(inout_matrix.data),
                static_cast<float*>(inout_eigvals.data)
            );
            break;
        case py::dtype::num_of<double>():
            status = magma_symmetric_solver_gpu<double>(
                solver_context,
                static_cast<double*>(inout_matrix.data),
                static_cast<double*>(inout_eigvals.data)
            );
            break;
        case py::dtype::num_of<std::complex<float>>():
            status = magma_hermitian_solver_gpu<float>(
                solver_context,
                static_cast<magmaComplex<float>*>(inout_matrix.data),
                static_cast<float*>(inout_eigvals.data)
            );
            break;
        case py::dtype::num_of<std::complex<double>>():
            status = magma_hermitian_solver_gpu<double>(
                solver_context,
                static_cast<magmaComplex<double>*>(inout_matrix.data),
                static_cast<double*>(inout_eigvals.data)
            );
            break;
    default:
        throw std::invalid_argument("Unsupported matrix dtype");
    }

    if (status < 0)
    {
        throw std::runtime_error("Invalid input to MAGMA solver at position " + std::to_string(-status));
    }
    else if (status > 0)
    {
        PyErr_WarnEx(PyExc_RuntimeWarning,
            "MAGMA eigensolver failed to converge",
            1
        );
    }
}

static void init_magma_internals()
{
    MAGMA_CHECK(magma_init());

    // Cache things like number of GPUs available to MAGMA
    magma_int_t ndevices;
    magma_device_t devices[ MagmaMaxGPUs ];
    magma_getdevices(devices, MagmaMaxGPUs, &ndevices);

    gpaw::magma_static_data.num_gpus = ndevices;
}

static void finalize_magma_internals()
{
    MAGMA_CHECK(magma_finalize());
}

} // namespace gpaw


bool bind_magma_submodule(pybind11::module_ gpu_module)
{
    if (!gpu_module || gpu_module == Py_None)
    {
        return false;
    }

    py::module_ m = py::reinterpret_borrow<py::module_>(gpu_module);
    if (m.is_none())
    {
        return false;
    }
    py::module_ submodule = m.def_submodule("magma", "MAGMA bindings for GPAW");

    submodule.def("magma_init", &gpaw::init_magma_internals,
    R"(Initializes MAGMA library. Must be called come AFTER any calls to cudaSetValidDevices and cudaSetDeviceFlags.
        Call only if GPUs are available.)"
    );

    submodule.def("magma_finalize", &gpaw::finalize_magma_internals,
        R"(Cleanup of internal MAGMA state.)"
    );

    submodule.def("eigh_magma_numpy", &gpaw::eigh_magma_numpy,
    py::arg("inout_matrix"), py::arg("inout_eigenvalues"),
    py::arg("uplo"), py::arg("num_gpus"),
    R"(
        Solves eigensystem on the GPU with Numpy input/output. This is an in-place solver:
        the input matrix will be overwritten with resulting eigenvectors. The input eigenvalue array
        must already be allocated to correct size (its contents don't matter).
        Input matrix is in Numpy/Scipy conventions. Output is still in MAGMA (Fortran) convention,
        so you will need to conjugate transpose to get back to Numpy conventions for eigenvectors.

        Passing num_gpus > 1 will instruct the solver to utilize multiple GPUs. This can be beneficial for large matrices (N >= 10k).
        Note though that the GPUs must be directly reachable from the same Cuda/HIP context, ie. this is a single-node solver without Scalapack-like support.)"
    );

    submodule.def("eigh_magma_cupy", &gpaw::eigh_magma_cupy,
        py::arg("inout_matrix"), py::arg("inout_eigenvalues"), py::arg("uplo"),
        R"(
        Solves eigensystem on the GPU with Cupy input/output. This is an in-place solver:
        the input matrix will be overwritten with resulting eigenvectors. The input eigenvalue array
        must already be allocated to correct size (its contents don't matter).
        Input matrix is in Cupy/Scipy conventions. Output is still in MAGMA (Fortran) convention,
        so you will need to conjugate transpose to get back to Numpy conventions for eigenvectors.)"
    );

    return true;
}
