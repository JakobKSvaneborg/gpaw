#include "array_life_support.hpp"
#include "pyarray_utils.hpp"
#include "template_utils.hpp"

#include <Python.h>
#include <mutex>

namespace gpaw
{

#ifdef GPAW_GPU_ARRAY_DEBUG
static int g_arrays_in_use = 0;
#endif

std::mutex g_pending_decrefs_mutex;
std::vector<PyObject*> g_pending_decrefs; // should this be volatile?!

CLINKAGE PyObject* flush_pending_decrefs(PyObject* self, PyObject* args)
{
    if (g_pending_decrefs.empty())
    {
        Py_RETURN_NONE;
    }

    std::vector<PyObject*> local_pending_decrefs;
    {
        std::lock_guard<std::mutex> lock(g_pending_decrefs_mutex);
        local_pending_decrefs.swap(g_pending_decrefs);
    }

    for (PyObject* obj : local_pending_decrefs)
    {
        Py_DECREF(obj);
    }

    Py_RETURN_NONE;
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

        std::lock_guard<std::mutex> lock(g_pending_decrefs_mutex);
        for (PyObject* obj : vec_copy)
        {
            g_pending_decrefs.push_back(obj);
        }
    };

    gpu_host_callback(stream, wrapper);
    borrowed_objects.clear();
}

} // namespace gpaw
