#include "../extensions.h"
#include "gpu.h"
#include "gpu-complex.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef GPAW_WITH_MAGMA
    #include "cpp/magma/magma_python_interface.h"
#endif

#include "lfc_gpu.h"

PyObject* gpaw_gpu_init(PyObject *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, ""))
        return NULL;

    bc_init_buffers_gpu();
    transformer_init_buffers_gpu();
    operator_init_buffers_gpu();
    reduce_init_buffers_gpu();
    lfc_reduce_init_buffers_gpu();
    blas_init_gpu();

#ifdef GPAW_WITH_MAGMA
    gpaw_magma_init();
#endif

    if (PyErr_Occurred())
        return NULL;
    else
        Py_RETURN_NONE;
}

PyObject* gpaw_gpu_delete(PyObject *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, ""))
        return NULL;

#ifdef GPAW_WITH_MAGMA
    gpaw_magma_finalize();
#endif

    reduce_dealloc_gpu();
    lfc_reduce_dealloc_gpu();
    bc_dealloc_gpu(1);
    transformer_dealloc_gpu(1);
    operator_dealloc_gpu(1);

    if (PyErr_Occurred())
        return NULL;
    else
        Py_RETURN_NONE;
}
