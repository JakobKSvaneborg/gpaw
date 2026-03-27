#pragma once

#include "python_utils.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "gpu/gpu_utils.h"


// Binds MAGMA wrappers as a submodule (_gpaw.gpu.magma)
bool bind_magma_submodule(pybind11::module_ gpu_module);
