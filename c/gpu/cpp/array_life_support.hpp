#pragma once

#include "../gpu-runtime.h"
#include "pyarray_utils.hpp"

#include <vector>

namespace gpaw
{

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
