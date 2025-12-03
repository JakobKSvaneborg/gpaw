#include "../../gpaw_utils.h"
#include "../gpu.h"
#include "../gpu-complex.h"
#include "pyarray_utils.hpp"
#include "array_life_support.hpp"
#include "../gpu_interface.h"
#include "pwlfc_expand.hpp"

static int get_dtype(PyObject* array)
{
    // Only these combinations are allowed. Make it so.
    // dtype.num 11      14        12      15
    // array     float32 complex64 float64 complex128

    int dtypenum = gpaw::Array_TYPE(array);
    assert(dtypenum == NP_FLOAT || dtypenum == NP_DOUBLE ||
           dtypenum == NP_FLOAT_COMPLEX || dtypenum == NP_DOUBLE_COMPLEX);
    return dtypenum;
}

static void assert_corresponding_real(int dtypenum, PyObject* array)
{
    // Only these combinations are allowed. Make it so.
    // dtypenum  11      14        12      15
    //           float32 complex64 float64 complex128
    //
    // realdtype 11      11        12      12
    // array     float32 float32   float64
    int realdtype = gpaw::Array_TYPE(array);
    assert((realdtype == NP_FLOAT && (dtypenum == NP_FLOAT || dtypenum == NP_FLOAT_COMPLEX)) ||
           (realdtype == NP_DOUBLE && (dtypenum == NP_DOUBLE || dtypenum == NP_DOUBLE_COMPLEX)));
    return;
}

PyObject* evaluate_lda_gpu(PyObject* self, PyObject* args)
{
    PyObject* n_obj;
    PyObject* v_obj;
    PyObject* e_obj;
    PyObject* stream_obj;
    if (!PyArg_ParseTuple(args, "OOOO",
                          &n_obj, &v_obj, &e_obj, &stream_obj))
        return NULL;

    gpuStream_t stream = (gpuStream_t) PyLong_AsVoidPtr(stream_obj);

    int nspin = gpaw::Array_DIM(n_obj, 0);
    if ((nspin != 1) && (nspin != 2))
    {
        PyErr_Format(PyExc_RuntimeError, "Expected 1 or 2 spins. Got %d.", nspin);
        return NULL;
    }
    int ng = 1;
    for (int d=1; d<gpaw::Array_NDIM(n_obj); d++)
    {
        ng *= gpaw::Array_DIM(n_obj, d);
    }

    gpaw::PyObjectPinner pinner;

    double* n_ptr = pinner.borrow_array_data<double>(n_obj);
    double* v_ptr = pinner.borrow_array_data<double>(v_obj);
    double* e_ptr = pinner.borrow_array_data<double>(e_obj);

    if (PyErr_Occurred())
    {
        return NULL;
    }

    pinner.commit();

    evaluate_lda_launch_kernel(nspin, ng,
                               n_ptr, v_ptr, e_ptr, stream);

    pinner.schedule_unpin(0);

    if (PyErr_Occurred())
    {
        return NULL;
    }

    Py_RETURN_NONE;
}

PyObject* evaluate_pbe_gpu(PyObject* self, PyObject* args)
{
    PyObject* n_obj;
    PyObject* v_obj;
    PyObject* sigma_obj;
    PyObject* dedsigma_obj;
    PyObject* e_obj;
    PyObject* stream_obj;
    if (!PyArg_ParseTuple(args, "OOOOOO",
                          &n_obj, &v_obj, &e_obj, &sigma_obj, &dedsigma_obj, &stream_obj))
        return NULL;
    int nspin = gpaw::Array_DIM(n_obj, 0);
    gpuStream_t stream = (gpuStream_t) PyLong_AsVoidPtr(stream_obj);

    if ((nspin != 1) && (nspin != 2))
    {
        PyErr_Format(PyExc_RuntimeError, "Expected 1 or 2 spins. Got %d.", nspin);
        return NULL;
    }
    int ng = 1;
    for (int d=1; d<gpaw::Array_NDIM(n_obj); d++)
    {
        ng *= gpaw::Array_DIM(n_obj, d);
    }

    gpaw::PyObjectPinner pinner;

    double* n_ptr = pinner.borrow_array_data<double>(n_obj);
    double* v_ptr = pinner.borrow_array_data<double>(v_obj);
    double* e_ptr = pinner.borrow_array_data<double>(e_obj);
    double* sigma_ptr = pinner.borrow_array_data<double>(sigma_obj);
    double* dedsigma_ptr = pinner.borrow_array_data<double>(dedsigma_obj);

    if (PyErr_Occurred())
    {
        return NULL;
    }

    pinner.commit();

    evaluate_pbe_launch_kernel(nspin, ng,
                               n_ptr,
                               v_ptr,
                               e_ptr,
                               sigma_ptr,
                               dedsigma_ptr,
                               stream);

    pinner.schedule_unpin(0);

    if (PyErr_Occurred())
    {
        return NULL;
    }

    Py_RETURN_NONE;
}

PyObject* dH_aii_times_P_ani_gpu(PyObject* self, PyObject* args)
{
    PyObject* dH_aii_obj;
    PyObject* ni_a_obj;
    PyObject* P_ani_obj;
    PyObject* outP_ani_obj;
    PyObject* stream_obj;
    if (!PyArg_ParseTuple(args, "OOOOO",
                          &dH_aii_obj, &ni_a_obj, &P_ani_obj, &outP_ani_obj, &stream_obj))
        return NULL;

    gpuStream_t stream = (gpuStream_t) PyLong_AsVoidPtr(stream_obj);
    if (gpaw::Array_DIM(ni_a_obj, 0) == 0)
    {
        Py_RETURN_NONE;
    }

    gpaw::PyObjectPinner pinner;

    void* dH_aii_dev = pinner.borrow_array_data<void>(dH_aii_obj);
    if (!dH_aii_dev)
    {
	PyErr_SetString(PyExc_RuntimeError, "Error in input dH_aii.");
        return NULL;
    }
    void* P_ani_dev = pinner.borrow_array_data<void>(P_ani_obj);
    if (!P_ani_dev)
    {
        PyErr_SetString(PyExc_RuntimeError, "Error in input P_ani.");
        return NULL;
    }
    void* outP_ani_dev = pinner.borrow_array_data<void>(outP_ani_obj);
    if (!outP_ani_dev)
    {
        PyErr_SetString(PyExc_RuntimeError, "Error in output outP_ani.");
        return NULL;
    }
    int32_t* ni_a = pinner.borrow_array_data<int32_t>(ni_a_obj);
    if (!ni_a)
    {
        PyErr_SetString(PyExc_RuntimeError, "Error in input ni_a.");
        return NULL;
    }

    int dtypenum = get_dtype(P_ani_obj);
    assert_corresponding_real(dtypenum, dH_aii_obj);
    assert(dtypenum == get_dtype(outP_ani_obj));

    assert(gpaw::Array_ITEMSIZE(ni_a_obj) == 4);

    int nA = gpaw::Array_DIM(ni_a_obj, 0);
    int nn = gpaw::Array_DIM(P_ani_obj, 0);
    int nI = gpaw::Array_DIM(P_ani_obj, 1);
    if (PyErr_Occurred())
    {
        return NULL;
    }

    pinner.commit();

    dH_aii_times_P_ani_launch_kernel(dtypenum, nA, nn, nI, ni_a, dH_aii_dev, P_ani_dev, outP_ani_dev, stream);

    pinner.schedule_unpin(0);

    if (PyErr_Occurred())
    {
        return NULL;
    }

    Py_RETURN_NONE;
}


PyObject* pwlfc_expand_gpu(PyObject* self, PyObject* args)
{
    PyObject *f_Gs_obj;
    PyObject *Gk_Gv_obj;
    PyObject *pos_av_obj;
    PyObject *eikR_a_obj;
    PyObject *Y_GL_obj;
    PyObject *l_s_obj;
    PyObject *a_J_obj;
    PyObject *s_J_obj;
    int cc;
    PyObject *f_GI_obj;
    PyObject *I_J_obj;
    PyObject *stream_obj;
    if (!PyArg_ParseTuple(args, "OOOOOOOOiOOO",
                          &f_Gs_obj, &Gk_Gv_obj, &pos_av_obj,
                          &eikR_a_obj, &Y_GL_obj,
                          &l_s_obj, &a_J_obj, &s_J_obj,
                          &cc, &f_GI_obj, &I_J_obj, &stream_obj)
    )
    {
        return NULL;
    }

    gpuStream_t stream = (gpuStream_t) PyLong_AsVoidPtr(stream_obj);

    gpaw::PyObjectPinner pinner;

    void *f_Gs = pinner.borrow_array_data<void>(f_Gs_obj);
    void *Y_GL = pinner.borrow_array_data<void>(Y_GL_obj);
    int *l_s = pinner.borrow_array_data<int>(l_s_obj);
    int *a_J = pinner.borrow_array_data<int>(a_J_obj);
    int *s_J = pinner.borrow_array_data<int>(s_J_obj);
    void *f_GI = pinner.borrow_array_data<void>(f_GI_obj);
    int nG = gpaw::Array_DIM(Gk_Gv_obj, 0);
    int *I_J = pinner.borrow_array_data<int>(I_J_obj);
    int nJ = gpaw::Array_DIM(a_J_obj, 0);
    int nL = gpaw::Array_DIM(Y_GL_obj, 1);
    int nI = gpaw::Array_DIM(f_GI_obj, 1);
    int natoms = gpaw::Array_DIM(pos_av_obj, 0);
    int nsplines = gpaw::Array_DIM(f_Gs_obj, 1);
    void* Gk_Gv = pinner.borrow_array_data<void>(Gk_Gv_obj);
    void* pos_av = pinner.borrow_array_data<void>(pos_av_obj);
    void* eikR_a = pinner.borrow_array_data<void>(eikR_a_obj);
    int dtype = get_dtype(f_GI_obj);
    if (PyErr_Occurred())
    {
        return NULL;
    }

    pinner.commit();

    pwlfc_expand_gpu_launch_kernel(dtype, f_Gs, Gk_Gv, pos_av, eikR_a, Y_GL,
                                   l_s, a_J, s_J, f_GI,
                                   I_J, nG, nJ, nL, nI, natoms, nsplines, cc, stream);

    pinner.schedule_unpin(0);

    if (PyErr_Occurred())
    {
        return NULL;
    }

    Py_RETURN_NONE;
}

PyObject* pw_insert_gpu(PyObject* self, PyObject* args)
{
    PyObject *c_nG_obj, *Q_G_obj, *tmp_nQ_obj, *stream_obj;
    double scale;
    int rx;
    int ry;
    int rz;
    if (!PyArg_ParseTuple(args, "OOdOiiiO",
                          &c_nG_obj, &Q_G_obj, &scale, &tmp_nQ_obj, &rx, &ry, &rz, &stream_obj)
    )
    {
        return NULL;
    }

    gpuStream_t stream = (gpuStream_t) PyLong_AsVoidPtr(stream_obj);

    gpaw::PyObjectPinner pinner;

    int32_t *Q_G = pinner.borrow_array_data<int32_t>(Q_G_obj);
    void *c_nG = pinner.borrow_array_data<void>(c_nG_obj);
    void *tmp_nQ = pinner.borrow_array_data<void>(tmp_nQ_obj);
    int nG = 0;
    int nQ = 0;
    int nb = 0;
    assert(gpaw::Array_NDIM(c_nG_obj) == gpaw::Array_NDIM(tmp_nQ_obj));
    if (gpaw::Array_NDIM(c_nG_obj) == 1)
    {
        nG = gpaw::Array_DIM(c_nG_obj, 0);
        nb = 1;
        nQ = gpaw::Array_DIM(tmp_nQ_obj, 0);
    }
    else
    {
        nG = gpaw::Array_DIM(c_nG_obj, 1);
        nb = gpaw::Array_DIM(c_nG_obj, 0);
        nQ = gpaw::Array_DIM(tmp_nQ_obj, 1);
    }
    if (PyErr_Occurred())
    {
        return NULL;
    }

    int dtypenum = get_dtype(c_nG_obj);
    assert(dtypenum == get_dtype(tmp_nQ_obj));

    pinner.commit();

    pw_insert_gpu_launch_kernel(dtypenum,
                                nb, nG, nQ,
                                c_nG,
                                Q_G,
                                scale,
                                tmp_nQ, rx, ry, rz, stream);

    pinner.schedule_unpin(0);

    if (PyErr_Occurred())
    {
        return NULL;
    }

    Py_RETURN_NONE;
}

PyObject* pw_norm_gpu(PyObject* self, PyObject* args)
{
    PyObject *result_x_obj, *C_xG_obj, *stream_obj;
    if (!PyArg_ParseTuple(args, "OOO",
                          &result_x_obj, &C_xG_obj, &stream_obj)
    )
    {
        return NULL;
    }

    gpaw::PyObjectPinner pinner;

    gpuStream_t stream = (gpuStream_t) PyLong_AsVoidPtr(stream_obj);

    void *result_x = pinner.borrow_array_data<void>(result_x_obj);
    void *C_xG = pinner.borrow_array_data<void>(C_xG_obj);
    int dtypenum = get_dtype(C_xG_obj);

    // Make sure number of dimensions are correct
    assert(gpaw::Array_NDIM(C_xG_obj) == 2);
    assert(gpaw::Array_NDIM(result_x_obj) == 1);

    // Make sure dtypes are correct
    assert_corresponding_real(dtypenum, result_x_obj);

    // Make sure dimensions match
    int nx = gpaw::Array_DIM(result_x_obj, 0);
    int nG = gpaw::Array_DIM(C_xG_obj, 1);
    assert(gpaw::Array_DIM(C_xG_obj, 0) == nx);

    if (PyErr_Occurred())
    {
        return NULL;
    }

    pinner.commit();

    pw_norm_gpu_launch_kernel(dtypenum,
                              nx, nG,
                              result_x,
                              C_xG, stream);

    pinner.schedule_unpin(0);

    if (PyErr_Occurred())
    {
        return NULL;
    }

    Py_RETURN_NONE;
}

PyObject* pw_norm_kinetic_gpu(PyObject* self, PyObject* args)
{
    PyObject *result_x_obj, *C_xG_obj, *kin_G_obj, *stream_obj;
    if (!PyArg_ParseTuple(args, "OOOO",
                          &result_x_obj, &C_xG_obj, &kin_G_obj, &stream_obj))
        return NULL;

    gpuStream_t stream = (gpuStream_t) PyLong_AsVoidPtr(stream_obj);

    gpaw::PyObjectPinner pinner;

    void *result_x = pinner.borrow_array_data<void>(result_x_obj);
    void *C_xG = pinner.borrow_array_data<void>(C_xG_obj);
    void *kin_G = pinner.borrow_array_data<void>(kin_G_obj);
    int dtypenum = get_dtype(C_xG_obj);

    // Make sure number of dimensions are correct
    assert(gpaw::Array_NDIM(C_xG_obj) == 2);
    assert(gpaw::Array_NDIM(result_x_obj) == 1);
    assert(gpaw::Array_NDIM(kin_G_obj) == 1);

    // Make sure dtypes are correct
    assert_corresponding_real(dtypenum, result_x_obj);
    assert_corresponding_real(dtypenum, kin_G_obj);

    // Make sure dimensions match
    int nx = gpaw::Array_DIM(result_x_obj, 0);
    int nG = gpaw::Array_DIM(C_xG_obj, 1);
    assert(gpaw::Array_DIM(kin_G_obj, 0) == nG);
    assert(gpaw::Array_DIM(C_xG_obj, 0) == nx);

    if (PyErr_Occurred())
    {
        return NULL;
    }

    pinner.commit();

    pw_norm_kinetic_gpu_launch_kernel(dtypenum,
                                      nx, nG,
                                      result_x,
                                      C_xG,
                                      kin_G, stream);

    pinner.schedule_unpin(0);

    if (PyErr_Occurred())
    {
        return NULL;
    }

    Py_RETURN_NONE;
}

PyObject* pw_amend_insert_realwf_gpu(PyObject* self, PyObject* args)
{
    PyObject *array_nQ_obj, *stream_obj;
    int n;
    int m;
    if (!PyArg_ParseTuple(args, "OiiO",
                          &array_nQ_obj, &n, &m, &stream_obj))
        return NULL;

    gpuStream_t stream = (gpuStream_t) PyLong_AsVoidPtr(stream_obj);

    gpaw::PyObjectPinner pinner;

    void *array_nQ = pinner.borrow_array_data<void>(array_nQ_obj);
    if (gpaw::Array_NDIM(array_nQ_obj) != 4)
    {
        PyErr_SetString(PyExc_RuntimeError, "array_nQ must be of (nb, NGx, NGy, NGz)-shape.");
        return NULL;
    }
    int nb = gpaw::Array_DIM(array_nQ_obj, 0);
    int nx = gpaw::Array_DIM(array_nQ_obj, 1);
    int ny = gpaw::Array_DIM(array_nQ_obj, 2);
    int nz = gpaw::Array_DIM(array_nQ_obj, 3);
    if (PyErr_Occurred())
    {
        return NULL;
    }
    int dtypenum = get_dtype(array_nQ_obj);

    pinner.commit();

    pw_amend_insert_realwf_gpu_launch_kernel(dtypenum, nb, nx, ny, nz, n, m, array_nQ, stream);

    pinner.schedule_unpin(0);

    if (PyErr_Occurred())
    {
        return NULL;
    }

    Py_RETURN_NONE;
}


PyObject* add_to_density_gpu(PyObject* self, PyObject* args)
{
    PyObject *f_n_obj, *psit_nR_obj, *rho_R_obj, *stream_obj;
    if (!PyArg_ParseTuple(args, "OOOO",
                          &f_n_obj, &psit_nR_obj, &rho_R_obj, &stream_obj))
        return NULL;

    gpuStream_t stream = (gpuStream_t) PyLong_AsVoidPtr(stream_obj);

    int dtypenum = get_dtype(psit_nR_obj);

    gpaw::PyObjectPinner pinner;

    double *f_n = pinner.borrow_array_data<double>(f_n_obj);
    void *psit_nR = pinner.borrow_array_data<void>(psit_nR_obj);
    double *rho_R = pinner.borrow_array_data<double>(rho_R_obj);
    int nb = gpaw::Array_SIZE(f_n_obj);
    int nR = gpaw::Array_SIZE(psit_nR_obj) / nb;

    // If running on same precision, then this should be the case
    // assert_corresponding_real(dtypenum, rho_R_obj);
    // However, we always have the density as double:
    assert(get_dtype(rho_R_obj) == NP_DOUBLE);

    if (PyErr_Occurred())
    {
        return NULL;
    }

    pinner.commit();

    add_to_density_gpu_launch_kernel(nb, nR, f_n, psit_nR, rho_R, dtypenum, stream);
    pinner.schedule_unpin(0);

    if (PyErr_Occurred())
    {
        return NULL;
    }

    Py_RETURN_NONE;
}


PyObject* calculate_residual_gpu(PyObject* self, PyObject* args)
{
    PyObject *residual_nG_obj, *eps_n_obj, *wf_nG_obj, *stream_obj;
    if (!PyArg_ParseTuple(args, "OOOO",
                          &residual_nG_obj, &eps_n_obj, &wf_nG_obj, &stream_obj))
        return NULL;

    gpuStream_t stream = (gpuStream_t) PyLong_AsVoidPtr(stream_obj);
    gpaw::PyObjectPinner pinner;
    void *residual_nG = pinner.borrow_array_data<void>(residual_nG_obj);
    void* eps_n = pinner.borrow_array_data<void>(eps_n_obj);
    void *wf_nG = pinner.borrow_array_data<void>(wf_nG_obj);
    int nn = gpaw::Array_DIM(residual_nG_obj, 0);
    int nG = 1;
    // nG is required to be below 2**31 here, which should be ok.
    for (int d=1; d<gpaw::Array_NDIM(residual_nG_obj); d++)
    {
        nG *= gpaw::Array_DIM(residual_nG_obj, d);
    }
    if (PyErr_Occurred())
    {
        return NULL;
    }
    int dtypenum = get_dtype(residual_nG_obj);
    assert_corresponding_real(dtypenum, eps_n_obj);

    pinner.commit();

    calculate_residual_launch_kernel(dtypenum, nG, nn, residual_nG, eps_n, wf_nG, stream);
    pinner.schedule_unpin(0);

    if (PyErr_Occurred())
    {
        return NULL;
    }

    Py_RETURN_NONE;
}
