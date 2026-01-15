#pragma once

// FIXME: can we forward declare the PyObject typedef to avoid including Python.h,
// because of their include order requirement mess? Should work in C++11. 
#include "../python_utils.h"
#include "../gpaw_utils.h"
#include "gpu_utils.h"

GPAW_GPU_LINKAGE_BEGIN

PyObject* gpaw_gpu_init(PyObject *self, PyObject *args);
PyObject* gpaw_gpu_delete(PyObject *self, PyObject *args);
PyObject* csign_gpu(PyObject *self, PyObject *args);
PyObject* scal_gpu(PyObject *self, PyObject *args);
PyObject* multi_scal_gpu(PyObject *self, PyObject *args);
PyObject* mmm_gpu(PyObject *self, PyObject *args);
PyObject* gemm_gpu(PyObject *self, PyObject *args);
PyObject* gemv_gpu(PyObject *self, PyObject *args);
PyObject* rk_gpu(PyObject *self, PyObject *args);
PyObject* axpy_gpu(PyObject *self, PyObject *args);
PyObject* multi_axpy_gpu(PyObject *self, PyObject *args);
PyObject* r2k_gpu(PyObject *self, PyObject *args);
PyObject* dotc_gpu(PyObject *self, PyObject *args);
PyObject* dotu_gpu(PyObject *self, PyObject *args);
PyObject* multi_dotu_gpu(PyObject *self, PyObject *args);
PyObject* multi_dotc_gpu(PyObject *self, PyObject *args);
PyObject* add_linear_field_gpu(PyObject *self, PyObject *args);
PyObject* elementwise_multiply_add_gpu(PyObject *self, PyObject *args);
PyObject* multi_elementwise_multiply_add_gpu(PyObject *self, PyObject *args);
PyObject* ax2py_gpu(PyObject *self, PyObject *args);
PyObject* multi_ax2py_gpu(PyObject *self, PyObject *args);
PyObject* axpbyz_gpu(PyObject *self, PyObject *args);
PyObject* axpbz_gpu(PyObject *self, PyObject *args);
PyObject* fill_gpu(PyObject *self, PyObject *args);
PyObject* pwlfc_expand_gpu(PyObject *self, PyObject *args);
PyObject* pw_insert_gpu(PyObject *self, PyObject *args);
PyObject* pw_norm_kinetic_gpu(PyObject *self, PyObject *args);
PyObject* pw_norm_gpu(PyObject *self, PyObject *args);

PyObject* pw_amend_insert_realwf_gpu(PyObject *self, PyObject *args);
PyObject* add_to_density_gpu(PyObject* self, PyObject* args);
PyObject* dH_aii_times_P_ani_gpu(PyObject* self, PyObject* args);
PyObject* evaluate_lda_gpu(PyObject* self, PyObject* args);
PyObject* evaluate_pbe_gpu(PyObject* self, PyObject* args);
PyObject* calculate_residual_gpu(PyObject* self, PyObject* args);

PyObject* flush_pending_decrefs(PyObject* self, PyObject* args);

GPAW_GPU_LINKAGE_END


