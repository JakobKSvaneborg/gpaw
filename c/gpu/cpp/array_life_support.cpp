#include "array_life_support.hpp"
#include "pyarray_utils.hpp"
#include "utils.hpp"
#include "gpu_core.hpp"
#include "../gpu_interface.h"

#include "../../python_utils.h"
#include <mutex>

PyObject* flush_pending_decrefs(PyObject* self, PyObject* args)
{
#ifdef GPAW_GPU_LIFETIME_GUARD
    if (life_support::g_pending_decrefs.empty())
    {
        Py_RETURN_NONE;
    }

    std::vector<PyObject*> local_pending_decrefs;
    {
        std::lock_guard<std::mutex> lock(life_support::g_pending_decrefs_mutex);
        local_pending_decrefs.swap(life_support::g_pending_decrefs);
    }

    for (PyObject* obj : local_pending_decrefs)
    {
        Py_DECREF(obj);
    }
#endif

    Py_RETURN_NONE;
}

namespace gpaw
{

namespace life_support
{
// Global cache for storing objects that have pending unpin/decref
#ifdef GPAW_GPU_LIFETIME_GUARD
static std::vector<PyObject*> g_pending_decrefs; // should this be volatile?!
static std::mutex g_pending_decrefs_mutex;
#endif
} // namespace life_support


PyObjectPinner::PyObjectPinner()
{
}

PyObjectPinner::PyObjectPinner(size_t reserve_count)
{
#ifdef GPAW_GPU_LIFETIME_GUARD
    objects.reserve(reserve_count);
#endif
}

void PyObjectPinner::commit()
{
#ifdef GPAW_GPU_ARRAY_DEBUG
    assert(!has_committed && "Can't commit object pinning twice");
    has_committed = true;
#endif
#ifdef GPAW_GPU_LIFETIME_GUARD
    for (PyObject* obj : objects)
    {
        Py_INCREF(obj);
    }
#endif
}

void PyObjectPinner::schedule_unpin(gpuStream_t stream)
{

#ifdef GPAW_GPU_ARRAY_DEBUG
    assert(has_committed && "You are calling schedule_unpin() without committing the pinning first");
#endif

#ifdef GPAW_GPU_LIFETIME_GUARD
    if (objects.empty())
    {
        return;
    }

    // Move the stored pointers to a lambda so the callback works even if this calling object has been destroyed
    auto wrapper = [vec_copy = std::move(objects)]() mutable
    {
        // Add our pointers to the global cache of pending unpins
        std::lock_guard<std::mutex> lock(life_support::g_pending_decrefs_mutex);
        for (PyObject* obj : vec_copy)
        {
            life_support::g_pending_decrefs.push_back(obj);
        }
    };

    gpu_host_callback(stream, wrapper);
#endif
}

} // namespace gpaw
