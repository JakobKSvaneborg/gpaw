#pragma once

/* Includes Python.h and Numpy headers with correct defines.
* Other files that need Python stuff should include this file instead of manually including Python or Numpy.
* Numpy array imports are done in _gpaw_so.c. */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API

// Array import should be done only from the source file that inits the GPAW module
#ifndef __GPAW_SHOULD_IMPORT_NUMPY
    #define NO_IMPORT_ARRAY
#endif

#include <numpy/arrayobject.h>
