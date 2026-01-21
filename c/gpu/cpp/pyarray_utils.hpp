#pragma once

// TODO: combine this with some common python header when moving to full C++.
// Would help with include order requirements from Python.h

#include "python_utils.h"
#include "gpu/gpu-runtime.h"
#include "utils.hpp"
#include "gpaw_utils.h"

#include <cstdint>
#include <cassert>
#include <type_traits>

// Utility functions for working with Python arrays.
// Needed when working with Cupy arrays in particular, which do not define a nice C-interface.
// For Numpy (PyArrayObject) these are wrappers around the Numpy Array API.
// Many of the Numpy built-in functions return integers as 'npy_intp,
//  which the docs define as "signed integer with same size as 'size_t'".
// Here we cast npy_intp to int64_t which should be more than enough for any practical size.
// For routines returning 'int' we often cast to 'int32_t' for explicity (with some exceptions)

namespace gpaw
{

// Returns true if the current thread has GIL
inline bool check_gil() noexcept
{
    return PyGILState_Check();
}

// Check that the current thread is holding GIL and abort on failure
#define ASSERT_GIL()    __assert_gil(__FILE__, __LINE__)
static inline void __assert_gil(const char *file, int line)
{
    if (!check_gil())
    {
        char msg[100];
        snprintf(msg, 100, "ENSURE_GIL() failed at %s:%d\n", file, line);
        gpaw_abort(msg);
    }
}

// RAII helper for scoped GIL acquisition
class GilGuard
{
public:
    GilGuard()
    {
        gil_state = PyGILState_Ensure();
    }

    ~GilGuard()
    {
        PyGILState_Release(gil_state);
    }

    GilGuard(const GilGuard&) = delete;
    GilGuard& operator=(const GilGuard&) = delete;

private:
    PyGILState_STATE gil_state;
};

inline bool is_complex_dtype(pybind11::dtype dtype) { return dtype.kind() == 'c'; }
/* Returns true if the input array has c_contiguous flag set.
Note that contiguous 1D arrays are both C- and F-contiguous. */
inline bool is_c_contiguous(pybind11::array arr) { return arr.flags() & pybind11::array::c_style; }

/* Checks if the input object is a Cupy array (cupy.ndarray).
Empty Cupy array is considered valid.
NOTE: Not quite foolproof: we assume that only Numpy and Cupy arrays are passed around
so this will return true for anything that looks like an ndarray but is NOT a Numpy array. */
bool is_cupy_array(PyObject* obj);

// See the `is_cupy_array(PyObject*)` overload for docstring
inline bool is_cupy_array(pybind11::handle obj) { return is_cupy_array(obj.ptr()); }

/* Cheap, type-erased wrapper around a Python-side GPU array (cupy.ndarray).
Does NOT take ownership or copy the data.
Defined as a hidden symbol to avoid subtle ABI issues due to exposing pybind11::dtype, see
https://pybind11.readthedocs.io/en/stable/faq.html#someclass-declared-with-greater-visibility-than-the-type-of-its-field-someclass-member-wattributes
*/
struct GPAW_HIDDEN_SYMBOL PyDeviceArray
{
    // Allow default construction for pybind11 type casting
    PyDeviceArray() noexcept;
    PyDeviceArray(PyObject* array);
    PyDeviceArray(pybind11::handle array);

    // Number of array dimensions
    size_t ndim() const { return shape.size(); }
    bool is_c_contiguous() const { return c_contiguous; }

    void* data = nullptr;
    pybind11::dtype dtype;
    bool c_contiguous;
    // Use int64_t for shape/strides for compatibility with standards like dlpack
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;

private:
    void from_cupy(pybind11::handle array);
    void from_cupy(PyObject* obj) { from_cupy(pybind11::handle(obj)); }

    // Let pybind11 type caster access private members
    friend class pybind11::detail::type_caster<gpaw::PyDeviceArray>;
};

// Old CPython-style stuff below

// Get pointer to the array data
template<typename T>
T* Array_DATA(PyObject* obj)
{
    static_assert(!std::is_pointer<T>::value, "T must be a scalar type");

    // Equivalent to obj.data.ptr
    PyObject* ndarray_data = PyObject_GetAttrString(obj, "data");
    if (ndarray_data == nullptr)
    {
        return nullptr;
    }

    PyObject* ptr_data = PyObject_GetAttrString(ndarray_data, "ptr");
    Py_DECREF(ndarray_data);
    if (ptr_data == nullptr)
    {
        return nullptr;
    }

    T* ptr = reinterpret_cast<T*>(PyLong_AsVoidPtr(ptr_data));
    Py_DECREF(ptr_data);
    return ptr;
}


bool Array_32BIT(PyObject* obj);

// Number of dimensions
int32_t Array_NDIM(PyObject* a);

// Size of d-th dimension
int64_t Array_DIM(PyObject* a, int32_t d);

// Size of array element in bytes
int64_t Array_ITEMSIZE(PyObject* a);

// Total number of elements in the array. See Array_NBYTES for total size in bytes
int64_t Array_SIZE(PyObject* a);

// Total number of bytes consumed by the array
int64_t Array_NBYTES(PyObject* a);

// Get built-in (Numpy) typenumber of elements of the array.
int Array_TYPE(PyObject* a);

// True if the array type is any complex floating point number
bool Array_ISCOMPLEX(PyObject* a);

//~
// Begin overloads for PyArrayObject, ie. Numpy array

template<typename T>
T* Array_DATA(PyArrayObject* a)
{
    return reinterpret_cast<T*>(PyArray_DATA(a));
}

int32_t Array_NDIM(PyArrayObject* a);
int64_t Array_DIM(PyArrayObject* a, int32_t d);
int64_t Array_ITEMSIZE(PyArrayObject* a);
int64_t Array_SIZE(PyArrayObject* a);
int64_t Array_NBYTES(PyArrayObject* a);
int Array_TYPE(PyArrayObject* a);
bool Array_ISCOMPLEX(PyArrayObject* a);
//~ End Numpy

} // namespace gpaw
