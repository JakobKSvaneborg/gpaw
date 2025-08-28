#include "pyarray_utils.hpp"
#include "template_utils.hpp"
namespace gpaw
{

#ifdef GPAW_GPU_ARRAY_DEBUG
    static int g_arrays_in_use = 0;
#endif

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


ArrayBorrowList::ArrayBorrowList()
{
}

ArrayBorrowList::ArrayBorrowList(size_t reserve_count)
{
    borrowed_objects.reserve(reserve_count);
}

void ArrayBorrowList::add(PyObject* obj)
{
    assert(Array_DATA<void>(obj) != nullptr && "Tried to borrow from invalid array");
    borrowed_objects.push_back(obj);
}

void ArrayBorrowList::commit()
{
    for (PyObject* obj : borrowed_objects)
    {
        Py_INCREF(obj);
    }
}

void ArrayBorrowList::flush()
{
    for (PyObject* obj : borrowed_objects)
    {
        Py_DECREF(obj);
    }
    borrowed_objects.clear();
}

void ArrayBorrowList::schedule_array_unuse(gpuStream_t stream)
{
    if (borrowed_objects.empty())
    {
        return;
    }

    auto wrapper = [vec_copy = std::move(borrowed_objects)]() mutable
    {
        for (PyObject* obj : vec_copy)
        {
            Py_DECREF(obj);
        }
        vec_copy.clear();
    };

    gpu_host_callback(stream, wrapper);
    borrowed_objects.clear();
}


} // namespace gpaw
