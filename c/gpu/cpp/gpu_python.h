#pragma once

#include "python_forward_declares.h"
#include "gpu/gpu_utils.h"

#ifndef __cplusplus
    #include <stdbool.h>
#endif

/* Creates _gpaw.gpu module and binds GPU stuff to it.
* EXPERIMENTAL: only some very new code is currently bound here.
*/
GPAW_GPU_LINKAGE bool bind_gpu_submodule(PyObject* module);
