#include "pyarray_utils.hpp"
#include "utils.hpp"

namespace gpaw
{

PyDeviceArray::PyDeviceArray() noexcept
    : data(nullptr), c_contiguous(false)
{
}

PyDeviceArray::PyDeviceArray(PyObject* array)
    : PyDeviceArray(pybind11::handle(array))
{
}

PyDeviceArray::PyDeviceArray(pybind11::handle array)
{
    if (!is_cupy_array(array))
    {
        throw std::invalid_argument("Not a CuPy ndarray");
    }
    from_cupy(array);
}

void PyDeviceArray::from_cupy(pybind11::handle array)
{
    // dtype
    pybind11::object dtype_obj = array.attr("dtype");
    pybind11::object num_obj = dtype_obj.attr("num");
    const int typenum = num_obj.cast<int>();
    dtype = dtype_from_typenum(typenum);

    // Read C-contiguity flag
    pybind11::object flags = array.attr("flags");
    c_contiguous = flags.attr("c_contiguous").cast<bool>();

    // Get data pointer
    data = reinterpret_cast<void*>(
        array.attr("data").attr("ptr").cast<std::uintptr_t>()
    );

    if (!data)
    {
        // This should always be a bug
        throw std::invalid_argument("Empty Cupy array passed to C++");
    }

    // This might be one extra copy?
    auto py_shape = array.attr("shape").cast<pybind11::tuple>();
    auto py_strides = array.attr("strides").cast<pybind11::tuple>();

    assert(py_shape.size() == py_strides.size());
    shape.resize(py_shape.size());
    strides.resize(py_strides.size());

    for (size_t i = 0; i < shape.size(); ++i)
    {
        shape[i] = py_shape[i].cast<int64_t>();
        strides[i] = py_strides[i].cast<int64_t>();
    }
}

bool is_cupy_array(PyObject* obj)
{
    if (!obj)
    {
        return false;
    }

    /* Fast path: check that obj is not a Numpy array and exposes dlpack interface .
    This is very naive, but will correctly identify Cupy arrays as long as we only work with Numpy/Cupy. */
    if (!PyArray_Check(obj) && PyObject_HasAttrString(obj, "__dlpack__"))
    {
        return true;
    }

    return false;
}


bool Array_32BIT(PyObject* obj)
{
    PyObject* index_32_bits = PyObject_GetAttrString(obj, "_index_32_bits");
    if (index_32_bits == nullptr)
    {
        return false; // PySetErr will have been called, just return
    }
    bool value = PyObject_IsTrue(index_32_bits);
    Py_DECREF(index_32_bits);
    return value;
}


int32_t Array_NDIM(PyObject* obj)
{
    // return len(obj.shape)
    PyObject* shape = PyObject_GetAttrString(obj, "shape");
    if (shape == NULL) return -1;
    const int32_t ndim = static_cast<int32_t>(PyTuple_Size(shape));
    Py_DECREF(shape);

    return ndim;
}

int64_t Array_DIM(PyObject* obj, int32_t dim)
{
    PyObject* shape = PyObject_GetAttrString(obj, "shape");
    if (shape == NULL) return -1;
    PyObject* pydim = PyTuple_GetItem(shape, dim);
    if (pydim == NULL)
    {
        Py_DECREF(shape);
        return -1;
    }

    const int64_t value = static_cast<int64_t>(PyLong_AS_LONG(pydim));

    Py_DECREF(shape);
    return value;
}

int64_t Array_ITEMSIZE(PyObject* obj)
{
    PyObject* dtype = PyObject_GetAttrString(obj, "dtype");
    if (dtype == NULL) return -1;
    PyObject* itemsize_obj = PyObject_GetAttrString(dtype, "itemsize");
    Py_DECREF(dtype);
    if (itemsize_obj == NULL) return -1;

    int64_t itemsize = static_cast<int64_t>(PyLong_AS_LONG(itemsize_obj));

    Py_DECREF(itemsize_obj);
    return itemsize;
}

int64_t Array_SIZE(PyObject* obj)
{
    PyObject* size = PyObject_GetAttrString(obj, "size");
    if (size == NULL) return -1;

    int64_t arraysize = static_cast<int64_t>(PyLong_AS_LONG(size));
    Py_DECREF(size);

    return arraysize;
}

int64_t Array_NBYTES(PyObject* obj)
{
    PyObject* nbytes = PyObject_GetAttrString(obj, "nbytes");
    if (nbytes == NULL) return -1;

    int64_t nbytesvalue = static_cast<int64_t>(PyLong_AS_LONG(nbytes));
    Py_DECREF(nbytes);

    return nbytesvalue;
}

int Array_TYPE(PyObject* obj)
{
    PyObject* dtype = PyObject_GetAttrString(obj, "dtype");
    if (dtype == NULL) return -1;

    PyObject* num = PyObject_GetAttrString(dtype, "num");
    Py_DECREF(dtype);

    if (num == NULL) return -1;

    int typenum = static_cast<int>(PyLong_AS_LONG(num));
    Py_DECREF(num);

    return typenum;
}

bool Array_ISCOMPLEX(PyObject* obj)
{
    return PyTypeNum_ISCOMPLEX(Array_TYPE(obj));
}


// Numpy overloads


bool Array_CHECK(PyArrayObject* a)
{
    return true;
}

int32_t Array_NDIM(PyArrayObject* a)
{
    return static_cast<int32_t>(PyArray_NDIM(a));
}

int64_t Array_DIM(PyArrayObject* a, int32_t d)
{
    return static_cast<int64_t>(PyArray_DIM(a, static_cast<int>(d)));
}

int64_t Array_ITEMSIZE(PyArrayObject* a)
{
    return PyArray_ITEMSIZE(a);
}

int64_t Array_SIZE(PyArrayObject* a)
{
    return PyArray_SIZE(a);
}

int64_t Array_NBYTES(PyArrayObject* a)
{
    return PyArray_NBYTES(a);
}

int Array_TYPE(PyArrayObject* a)
{
    return PyArray_TYPE(a);
}

bool Array_ISCOMPLEX(PyArrayObject* a)
{
    return PyArray_ISCOMPLEX(a);
}

} // namespace gpaw
