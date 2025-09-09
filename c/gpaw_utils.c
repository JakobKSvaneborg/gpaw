#include "gpaw_utils.h"

#include <Python.h>

void gpaw_set_runtime_error(const char* err_msg)
{
    PyErr_SetString(PyExc_RuntimeError, err_msg);
}
