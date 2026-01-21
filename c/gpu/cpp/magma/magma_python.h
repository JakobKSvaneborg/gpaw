#pragma once

#include "python_utils.h"
#include "gpu/gpu_utils.h"


// Binds MAGMA wrappers as a submodule (_gpaw.gpu.magma)
bool bind_magma_submodule(pybind11::module_ gpu_module);

/* Initializes MAGMA library. Must be called come AFTER any calls to cudaSetValidDevices
* and cudaSetDeviceFlags. Call only if GPUs are available.
*/
GPAW_GPU_LINKAGE void gpaw_magma_init();
GPAW_GPU_LINKAGE void gpaw_magma_finalize();
