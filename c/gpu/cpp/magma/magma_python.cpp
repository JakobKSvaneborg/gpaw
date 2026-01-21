#include "magma_python.h"
#include "gpu/cpp/pyarray_utils.hpp"
#include "magma_gpaw.hpp"
#include "gpaw_utils.h"

#include <cassert>

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
static magma_uplo_t get_magma_uplo(char* in_uplo_str)
{
    assert((strcmp(in_uplo_str, "L") == 0 || strcmp(in_uplo_str, "U") == 0)
        && "Invalid UPLO");

    return strcmp(in_uplo_str, "L") == 0 ? MagmaUpper : MagmaLower;
}

} // namespace gpaw

void gpaw_magma_init()
{
    MAGMA_CHECK(magma_init());

    // Cache things like number of GPUs available to MAGMA
    magma_int_t ndevices;
    magma_device_t devices[ MagmaMaxGPUs ];
    magma_getdevices(devices, MagmaMaxGPUs, &ndevices);

    gpaw::magma_static_data.num_gpus = ndevices;
}

void gpaw_magma_finalize()
{
    MAGMA_CHECK(magma_finalize());
}

bool bind_magma_submodule(pybind11::module_ gpu_module)
{
    namespace py = pybind11;

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

    return true;
}
