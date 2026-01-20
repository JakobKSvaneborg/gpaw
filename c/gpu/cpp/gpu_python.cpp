#include "python_utils.h"
#include "gpu_python.h"

#include "gpu/cpp/pyarray_utils.hpp"
#include "gpu/cpp/pybind_cupy_type_caster.hpp"

namespace py = pybind11;

bool bind_gpu_submodule(PyObject* module)
{
    if (!module || module == Py_None)
    {
        return false;
    }

    py::module_ m = py::reinterpret_borrow<py::module_>(module);
    if (m.is_none())
    {
        return false;
    }

    py::module_ submodule = m.def_submodule("gpu", "GPU specific C++ bindings for GPAW. (EXPERIMENTAL)");

    return true;
}
