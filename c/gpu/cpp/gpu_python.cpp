#include "python_utils.h"
#include "gpu_python.h"

#include "gpu/cpp/pyarray_utils.hpp"
#include "gpu/cpp/pybind_cupy_type_caster.hpp"

#ifdef GPAW_WITH_MAGMA
    #include "gpu/cpp/magma/magma_python.h"
#endif

namespace py = pybind11;

namespace gpaw
{

/* Bind this and call from Python with the same cupy array passed to as both args.
* Will test that array metadata is copied correctly. Throws on failure */
static void test_array_metadata(py::handle cupy_array, const PyDeviceArray& same_array)
{
    if (!is_cupy_array(cupy_array))
    {
        throw std::invalid_argument("Input does not look like a Cupy array");
    }

    py::object dtype_obj = cupy_array.attr("dtype");
    py::object num_obj = dtype_obj.attr("num");
    py::dtype dtype(num_obj.cast<int>());
    if (dtype != same_array.dtype)
    {
        throw std::runtime_error("dtype does not match");
    }

    py::object flags = cupy_array.attr("flags");
    const bool c_contiguous = flags.attr("c_contiguous").cast<bool>();
    if (c_contiguous != same_array.c_contiguous)
    {
        throw std::runtime_error("c_contiguous flag does not match");
    }

    // Get data pointer
    const std::uintptr_t data = cupy_array.attr("data").attr("ptr").cast<std::uintptr_t>();
    if (data != reinterpret_cast<uintptr_t>(same_array.data))
    {
        throw std::runtime_error("data pointer does not match");
    }

    auto py_shape = cupy_array.attr("shape").cast<pybind11::tuple>();
    auto py_strides = cupy_array.attr("strides").cast<pybind11::tuple>();

    // Lengths of shape and stride tuples
    if (py_shape.size() != same_array.shape.size())
    {
        throw std::runtime_error("dimensionality does not match");
    }
    if (py_strides.size() != same_array.strides.size())
    {
        throw std::runtime_error("stride array size does not match");
    }

    // Check that each element of shape/stride matches
    for (size_t i = 0; i < same_array.shape.size(); ++i)
    {
        if (same_array.shape[i] != py_shape[i].cast<int64_t>())
        {
            throw std::runtime_error("shape does not match at index " + std::to_string(i));
        }
    }

    for (size_t i = 0; i < same_array.strides.size(); ++i)
    {
        if (same_array.strides[i] != py_strides[i].cast<int64_t>())
        {
            throw std::runtime_error("stride does not match at index " + std::to_string(i));
        }
    }
}

} // namespace gpaw

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

    // Bind Cupy conversion testers (call from Python-side test)

    submodule.def(
        "test_cupy_input",
        [](const gpaw::PyDeviceArray& arr) {},
        py::arg("a"),
        R"(Simply tests that our type caster works, ie. that we can pass a Cupy ndarray to a C++ function accepting gpaw::PyDeviceArray)"
    );
    submodule.def("test_array_metadata",
        &gpaw::test_array_metadata,
        py::arg("array"), py::arg("same_array"),
        R"(Tests that the array type caster correctly copies metadata from cupy ndarray. Use by passing the same array as both inputs. Raises RuntimeError on failure.)"
    );

    bool submodules_ok = true;
#ifdef GPAW_WITH_MAGMA
    // creates gpaw.gpu.magma
    submodules_ok &= bind_magma_submodule(submodule);
#endif

    return submodules_ok;
}
