#pragma once

#include "../gpu-runtime.h"

#include <Python.h>

// gpaw_so.c handles array importing at the module level (needed for proper numpy init),
// so don't do it here again.
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include <cstdint>
#include <cassert>
#include <vector>

// Utility functions for working with Python arrays.
// Needed when working with Cupy arrays in particular, which do not define a nice C-interface.
// For Numpy (PyArrayObject) these are wrappers around the Numpy Array API.
// Many of the Numpy built-in functions return integers as 'npy_intp,
//  which the docs define as "signed integer with same size as 'size_t'".
// Here we cast npy_intp to int64_t which should be more than enough for any practical size.
// For routines returning 'int' we often cast to 'int32_t' for explicity (with some exceptions)

namespace gpaw
{

// Get pointer to the array data
template<typename T>
T* Array_DATA(PyObject* obj)
{
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

    T* ptr = reinterpret_cast<T*>(PyLong_AS_LONG(ptr_data));
    Py_DECREF(ptr_data);
    return ptr;
}

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


class ArrayBorrowList
{
public:
    ArrayBorrowList();
    ArrayBorrowList(size_t reserve_count);
    void add(PyObject* obj);
    // "Commits" the borrowing. This is where all stored objects get their ref counts increased
    void commit();
    void flush();
    void schedule_array_unuse(gpuStream_t stream);

protected:
    std::vector<PyObject*> borrowed_objects;
};

template<typename T>
T* borrow_array(PyObject* obj, ArrayBorrowList& borrow_list)
{
    T* data = Array_DATA<T>(obj);
    if (data)
    {
        borrow_list.add(obj);
    }
    return data;
}

} // namespace gpaw
