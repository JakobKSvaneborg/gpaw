#pragma once

#include "../gpu-runtime.h"
#include "pyarray_utils.hpp"

#include <vector>

namespace gpaw
{


/* Class PyObjectPinner - used to keep Python objects alive for as long as GPU
kernels are using them. Intended for Cupy arrays specifically.
"Pinning" means that all PyObjects registered with this pinner instance get
their reference count increased. The pinning must be undone at a later time
to avoid memory leaks. */
class PyObjectPinner
{
public:
    PyObjectPinner();
    PyObjectPinner(size_t reserve_count);

    // "Commits" the pinning. This is where all stored objects get their ref counts increased
    void commit();
    /* Schedules unpinning in a GPU stream. Call this after launching your kernel(s).
    This way the arrays will be unpinned once the kernel(s) finish executing.
    The pinner object does not need to be alive when the unpinning actually happens. */
    void schedule_unpin(gpuStream_t stream);

    /* Request to 'borrow' array data from a PyObject. The object is assumed to be
    an array, so that gpaw::Array_DATA() returns a valid pointer.
    This function returns that pointer while also registering the PyObject
    for life support. */
    template<typename T>
    T* borrow_array_data(PyObject* obj)
    {
        T* data = Array_DATA<T>(obj);
    #ifdef GPAW_GPU_ARRAY_DEBUG
        assert(data && "Not a Python array object");
    #endif
        if (data)
        {
            objects.push_back(obj);
        }
        return data;
    }

protected:
    std::vector<PyObject*> objects;

#ifdef GPAW_GPU_ARRAY_DEBUG
    bool has_committed = false;
#endif
};

} // namespace gpaw
