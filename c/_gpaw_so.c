// Does the actual init of GPAW module. Also needs to do Numpy array import,
// see https://numpy.org/doc/stable/reference/c-api/array.html#including-and-importing-the-c-api

#define __GPAW_SHOULD_IMPORT_NUMPY

#include "python_utils.h"
#include "_gpaw.h"

PyMODINIT_FUNC PyInit__gpaw(void)
{
    import_array1(0);
    return moduleinit();
}
