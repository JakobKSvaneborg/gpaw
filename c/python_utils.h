#pragma once

/* Includes Python.h and Numpy headers with correct defines.
* Other files that need Python stuff should include this file instead of manually including Python or Numpy.
* Numpy array imports are done in _gpaw_so.c. */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

/* NOTE: Numpy 2.0 dev version had a bug for a short period where its public headers
* literally did "#undef I". This is obviously horrible since I is defined in the C99 standard.
* See https://github.com/numpy/numpy/pull/26789.
* We attempt to re-define it here in case someone happens to be using this buggy version of Numpy.
* Downside is that this header now needs to include the complex number header(s).
*/
#include "gpaw_complex.h"

// Array import should be done only from the source file that inits the GPAW module
#ifndef __GPAW_SHOULD_IMPORT_NUMPY
    #define NO_IMPORT_ARRAY
#endif

#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#include <numpy/arrayobject.h>

#ifndef __cplusplus
    // Fix for numpy's undef I
    #ifndef I
        #define I _Complex_I
    #endif
#endif
