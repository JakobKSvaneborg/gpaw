#pragma once

// TODO: combine this with some common python header when moving to full C++.
// Would help with include order requirements from Python.h

#include "python_utils.h"
#include "gpu/gpu-runtime.h"

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

// Scoped numpy datatypes
enum class DataType
{
    eUnknown = -1,
    eBool = NPY_BOOL,
    eInt8 = NPY_INT8,
    eUint8 = NPY_UINT8,
    eInt16 = NPY_INT16,
    eUint16 = NPY_UINT16,
    eInt32 = NPY_INT32,
    eUint32 = NPY_UINT32,
    eInt64 = NPY_INT64,
    eUint64 = NPY_UINT64,
    eFloat32 = NPY_FLOAT32,
    eFloat64 = NPY_FLOAT64,
    eComplex64 = NPY_COMPLEX64,
    eComplex128 = NPY_COMPLEX128
};

// Map runtime DataType to compile-time T
template<typename T>
constexpr bool matches_dtype(DataType dtype)
{
    if constexpr (std::is_same_v<T, bool>) return dtype == DataType::eBool;
    if constexpr (std::is_same_v<T, int8_t>) return dtype == DataType::eInt8;
    if constexpr (std::is_same_v<T, uint8_t>) return dtype == DataType::eUint8;
    if constexpr (std::is_same_v<T, int16_t>) return dtype == DataType::eInt16;
    if constexpr (std::is_same_v<T, uint16_t>) return dtype == DataType::eUint16;
    if constexpr (std::is_same_v<T, npy_int32>) return dtype == DataType::eInt32;
    if constexpr (std::is_same_v<T, int32_t>) return dtype == DataType::eInt32;
    if constexpr (std::is_same_v<T, uint32_t>) return dtype == DataType::eUint32;
    if constexpr (std::is_same_v<T, int64_t>) return dtype == DataType::eInt64;
    if constexpr (std::is_same_v<T, uint64_t>) return dtype == DataType::eUint64;
    if constexpr (std::is_same_v<T, float>) return dtype == DataType::eFloat32;
    if constexpr (std::is_same_v<T, double>) return dtype == DataType::eFloat64;
    if constexpr (std::is_same_v<T, std::complex<float>>) return dtype == DataType::eComplex64;
    if constexpr (std::is_same_v<T, std::complex<double>>) return dtype == DataType::eComplex128;
    if constexpr (std::is_same_v<T, gpuFloatComplex>) return dtype == DataType::eComplex64;
    if constexpr (std::is_same_v<T, gpuDoubleComplex>) return dtype == DataType::eComplex128;
    else
    {
        // FIXME use static_assert, but looks like some trickery is needed
        assert("No match for dtype");
    }
    return false;
}

/* Convert Numpy's type identifier to a DataType enum.
For example, dtype_from_typenum(NPY_FLOAT32) would return DataType::eFloat32 */
constexpr inline DataType dtype_from_typenum(int typenum)
{
    return static_cast<DataType>(typenum);
}

/* Convert DataType to raw Numpy typenum */
constexpr inline int typenum_from_dtype(DataType dtype)
{
    return static_cast<int>(dtype);
}

inline bool is_complex_dtype(pybind11::dtype dtype) { return dtype.kind() == 'c'; }
// FIXME c_style has different semantics for 1D arrays
inline bool is_c_contiguous(pybind11::array arr) { return arr.flags() & pybind11::array::c_style; }

/* Checks if the input object is a Cupy array (cupy.ndarray).
Empty Cupy array is considered valid.
NOTE: Not quite foolproof: we assume that only Numpy and Cupy arrays are passed around
so this will return true for anything that looks like an ndarray but is NOT a Numpy array. */
bool is_cupy_array(PyObject* obj);

// See the `is_cupy_array(PyObject*)` overload for docstring
inline bool is_cupy_array(pybind11::handle obj) { return is_cupy_array(obj.ptr()); }

/* Cheap, type-erased wrapper around a Python-side GPU array (cupy.ndarray).
Does NOT take ownership or copy the data. */
struct PyDeviceArray
{
    // Allow default construction for pybind11 type casting
    PyDeviceArray() noexcept;
    PyDeviceArray(PyObject* array);
    PyDeviceArray(pybind11::handle array);

    void* data = nullptr;
    DataType dtype;
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
