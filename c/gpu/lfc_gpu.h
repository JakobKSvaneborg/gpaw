#pragma once

#include "gpu_interface.h"
#include "../lfc.h"

GPAW_GPU_LINKAGE_BEGIN

void bc_init_buffers_gpu();
void blas_init_gpu();
void transformer_init_buffers_gpu();
void operator_init_buffers_gpu();
void reduce_init_buffers_gpu();
void lfc_reduce_init_buffers_gpu();
void bc_dealloc_gpu(int force);
void transformer_dealloc_gpu(int force);
void operator_dealloc_gpu(int force);
void reduce_dealloc_gpu();
void lfc_reduce_dealloc_gpu();

void lfc_dealloc_gpu(LFCObject *self);
PyObject* NewLFCObject_gpu(LFCObject *self, PyObject *args);
PyObject* add_gpu(LFCObject *self, PyObject *args);
PyObject* integrate_gpu(LFCObject *self, PyObject *args);

GPAW_GPU_LINKAGE_END
