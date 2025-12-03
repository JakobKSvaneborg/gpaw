#pragma once

#include "../gpaw_utils.h"
#include "../python_utils.h"

/* Declarations of C++ functions in the GPU code that need to interface with the main GPAW extension.
* Old builds: Main GPAW is C code, GPU code is C++ => GPU functions called from GPAW extension need C-linkage
* New builds: All of GPAW is C++ => No need for C-linkage in GPU code either.
* To handle both options we define GPAW_GPU_LINKAGE that sets the correct linkage for GPU functions.
* No need to specify the linkage again when defining the function, assuming this header gets included in the file that defines it.
*
* TODO: When removing the old C build path, simply remove the definition of GPAW_GPU_LINKAGE.
*/

#ifdef GPAW_CPP
    // Everything is C++, no need for C-linkage anywhere
    #define GPAW_GPU_LINKAGE
    #define GPAW_GPU_LINKAGE_BEGIN
    #define GPAW_GPU_LINKAGE_END
#else
    // Main GPAW is C-code, need C-linkage for GPU functions that are called from it
    #define GPAW_GPU_LINKAGE        CLINKAGE
    #define GPAW_GPU_LINKAGE_BEGIN  CLINKAGE_BEGIN
    #define GPAW_GPU_LINKAGE_END    CLINKAGE_END
#endif

GPAW_GPU_LINKAGE_BEGIN
void gpaw_device_synchronize();

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


