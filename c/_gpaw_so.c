#include "_gpaw.h"

PyMODINIT_FUNC PyInit__gpaw(void)
{
    import_array1(0);
    return moduleinit();
}
