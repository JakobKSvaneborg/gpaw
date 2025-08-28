#include "../../gpaw_utils.h"

#include "../gpu.h"
#include "../gpu-complex.h"

#include "pyarray_utils.hpp"

CLINKAGE
{
void calculate_residual_launch_kernel(int dtypenum,
                                      int nG,
                                      int nn,
                                      void* residual_ng,
                                      void* eps_n,
                                      void* wf_nG);

void pwlfc_expand_gpu_launch_kernel(int dtypenum,
                                    void* f_Gs,
                                    void* Gk_Gv,
                                    void* pos_av,
                                    void* eikR_a,
                                    void* Y_GL,
                                    int* l_s,
                                    int* a_J,
                                    int* s_J,
                                    void* f_GI,
                                    int* I_J,
                                    int nG,
                                    int nJ,
                                    int nL,
                                    int nI,
                                    int natoms,
                                    int nsplines,
                                    bool cc);

void pw_insert_gpu_launch_kernel(
                             int dtypenum,
                             int nb,
                             int nG,
                             int nQ,
                             void* c_nG,
                             npy_int32* Q_G,
                             double scale,
                             void* tmp_nQ,
                             int rx, int ry, int rz);

void pw_norm_gpu_launch_kernel(int dtypenum,
                               int nx, int nG,
                               void* result_x,
                               void* C_xG);

void pw_norm_kinetic_gpu_launch_kernel(int dtypenum,
                                       int nx, int nG,
                                       void* result_x,
                                       void* C_xG,
                                       void* kin_G);

void pw_amend_insert_realwf_gpu_launch_kernel(int dtypenum,
                                              int nb,
                                              int nx,
                                              int ny,
                                              int nz,
                                              int n,
                                              int m,
                                              void* array_nQ);

void add_to_density_gpu_launch_kernel(int nb,
                                      int nR,
                                      void* f_n,
                                      void* psit_nR,
                                      void* rho_R,
                                      int dtypenum);


void dH_aii_times_P_ani_launch_kernel(int dtypenum,
                                      int nA, int nn,
                                      int nI, npy_int32* ni_a,
                                      void* dH_aii_dev,
                                      void* P_ani_dev,
                                      void* outP_ani_dev);

void evaluate_pbe_launch_kernel(int nspin, int ng,
                                double* n,
                                double* v,
                                double* e,
                                double* sigma,
                                double* dedsigma);

void evaluate_lda_launch_kernel(int nspin, int ng,
                                double* n,
                                double* v,
                                double* e);

} // CLINKAGE

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

CLINKAGE PyObject* evaluate_lda_gpu(PyObject* self, PyObject* args)
{
    PyObject* n_obj;
    PyObject* v_obj;
    PyObject* e_obj;
    if (!PyArg_ParseTuple(args, "OOO",
                          &n_obj, &v_obj, &e_obj))
        return NULL;
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
    double* n_ptr = gpaw::Array_DATA<double>(n_obj);
    double* v_ptr = gpaw::Array_DATA<double>(v_obj);
    double* e_ptr = gpaw::Array_DATA<double>(e_obj);
    if (PyErr_Occurred())
    {
        return NULL;
    }
    evaluate_lda_launch_kernel(nspin, ng,
                               n_ptr, v_ptr, e_ptr);
    Py_RETURN_NONE;
}

CLINKAGE PyObject* evaluate_pbe_gpu(PyObject* self, PyObject* args)
{
    PyObject* n_obj;
    PyObject* v_obj;
    PyObject* sigma_obj;
    PyObject* dedsigma_obj;
    PyObject* e_obj;
    if (!PyArg_ParseTuple(args, "OOOOO",
                          &n_obj, &v_obj, &e_obj, &sigma_obj, &dedsigma_obj))
        return NULL;
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
    double* n_ptr = gpaw::Array_DATA<double>(n_obj);
    double* v_ptr = gpaw::Array_DATA<double>(v_obj);
    double* e_ptr = gpaw::Array_DATA<double>(e_obj);
    double* sigma_ptr = gpaw::Array_DATA<double>(sigma_obj);
    double* dedsigma_ptr = gpaw::Array_DATA<double>(dedsigma_obj);
    if (PyErr_Occurred())
    {
        return NULL;
    }
    evaluate_pbe_launch_kernel(nspin, ng,
                               n_ptr,
                               v_ptr,
                               e_ptr,
                               sigma_ptr,
                               dedsigma_ptr);
    Py_RETURN_NONE;
}

CLINKAGE PyObject* dH_aii_times_P_ani_gpu(PyObject* self, PyObject* args)
{
    PyObject* dH_aii_obj;
    PyObject* ni_a_obj;
    PyObject* P_ani_obj;
    PyObject* outP_ani_obj;

    if (!PyArg_ParseTuple(args, "OOOO",
                          &dH_aii_obj, &ni_a_obj, &P_ani_obj, &outP_ani_obj))
        return NULL;


    if (gpaw::Array_DIM(ni_a_obj, 0) == 0)
    {
        Py_RETURN_NONE;
    }

    void* dH_aii_dev = gpaw::Array_DATA<void>(dH_aii_obj);
    if (!dH_aii_dev)
    {
	PyErr_SetString(PyExc_RuntimeError, "Error in input dH_aii.");
        return NULL;
    }
    void* P_ani_dev = gpaw::Array_DATA<void>(P_ani_obj);
    if (!P_ani_dev)
    {
        PyErr_SetString(PyExc_RuntimeError, "Error in input P_ani.");
        return NULL;
    }
    void* outP_ani_dev = gpaw::Array_DATA<void>(outP_ani_obj);
    if (!outP_ani_dev)
    {
        PyErr_SetString(PyExc_RuntimeError, "Error in output outP_ani.");
        return NULL;
    }
    npy_int32* ni_a = gpaw::Array_DATA<npy_int32>(ni_a_obj);
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

    dH_aii_times_P_ani_launch_kernel(dtypenum, nA, nn, nI, ni_a, dH_aii_dev, P_ani_dev, outP_ani_dev);
    Py_RETURN_NONE;
}


CLINKAGE PyObject* pwlfc_expand_gpu(PyObject* self, PyObject* args)
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

    if (!PyArg_ParseTuple(args, "OOOOOOOOiOO",
                          &f_Gs_obj, &Gk_Gv_obj, &pos_av_obj,
                          &eikR_a_obj, &Y_GL_obj,
                          &l_s_obj, &a_J_obj, &s_J_obj,
                          &cc, &f_GI_obj, &I_J_obj))
        return NULL;
    void *f_Gs = gpaw::Array_DATA<void>(f_Gs_obj);
    void *Y_GL = gpaw::Array_DATA<void>(Y_GL_obj);
    int *l_s = gpaw::Array_DATA<int>(l_s_obj);
    int *a_J = gpaw::Array_DATA<int>(a_J_obj);
    int *s_J = gpaw::Array_DATA<int>(s_J_obj);
    void *f_GI = gpaw::Array_DATA<void>(f_GI_obj);
    int nG = gpaw::Array_DIM(Gk_Gv_obj, 0);
    int *I_J = gpaw::Array_DATA<int>(I_J_obj);
    int nJ = gpaw::Array_DIM(a_J_obj, 0);
    int nL = gpaw::Array_DIM(Y_GL_obj, 1);
    int nI = gpaw::Array_DIM(f_GI_obj, 1);
    int natoms = gpaw::Array_DIM(pos_av_obj, 0);
    int nsplines = gpaw::Array_DIM(f_Gs_obj, 1);
    void* Gk_Gv = gpaw::Array_DATA<void>(Gk_Gv_obj);
    void* pos_av = gpaw::Array_DATA<void>(pos_av_obj);
    void* eikR_a = gpaw::Array_DATA<void>(eikR_a_obj);
    int dtype = get_dtype(f_GI_obj);
    if (PyErr_Occurred())
    {
        return NULL;
    }
    pwlfc_expand_gpu_launch_kernel(dtype, f_Gs, Gk_Gv, pos_av, eikR_a, Y_GL,
                                   l_s, a_J, s_J, f_GI,
                                   I_J, nG, nJ, nL, nI, natoms, nsplines, cc);
    Py_RETURN_NONE;
}

CLINKAGE PyObject* pw_insert_gpu(PyObject* self, PyObject* args)
{
    PyObject *c_nG_obj, *Q_G_obj, *tmp_nQ_obj;
    double scale;
    int rx;
    int ry;
    int rz;
    if (!PyArg_ParseTuple(args, "OOdOiii",
                          &c_nG_obj, &Q_G_obj, &scale, &tmp_nQ_obj, &rx, &ry, &rz))
        return NULL;
    npy_int32 *Q_G = gpaw::Array_DATA<npy_int32>(Q_G_obj);
    void *c_nG = gpaw::Array_DATA<void>(c_nG_obj);
    void *tmp_nQ = gpaw::Array_DATA<void>(tmp_nQ_obj);
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

    pw_insert_gpu_launch_kernel(dtypenum,
                                nb, nG, nQ,
                                c_nG,
                                Q_G,
                                scale,
                                tmp_nQ, rx, ry, rz);
    Py_RETURN_NONE;
}

CLINKAGE PyObject* pw_norm_gpu(PyObject* self, PyObject* args)
{
    PyObject *result_x_obj, *C_xG_obj;
    if (!PyArg_ParseTuple(args, "OO",
                          &result_x_obj, &C_xG_obj))
        return NULL;

    void *result_x = gpaw::Array_DATA<void>(result_x_obj);
    void *C_xG = gpaw::Array_DATA<void>(C_xG_obj);
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
    pw_norm_gpu_launch_kernel(dtypenum,
                              nx, nG,
                              result_x,
                              C_xG);
    Py_RETURN_NONE;
}

CLINKAGE PyObject* pw_norm_kinetic_gpu(PyObject* self, PyObject* args)
{
    PyObject *result_x_obj, *C_xG_obj, *kin_G_obj;
    if (!PyArg_ParseTuple(args, "OOO",
                          &result_x_obj, &C_xG_obj, &kin_G_obj))
        return NULL;

    void *result_x = gpaw::Array_DATA<void>(result_x_obj);
    void *C_xG = gpaw::Array_DATA<void>(C_xG_obj);
    void *kin_G = gpaw::Array_DATA<void>(kin_G_obj);
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
    pw_norm_kinetic_gpu_launch_kernel(dtypenum,
                                      nx, nG,
                                      result_x,
                                      C_xG,
                                      kin_G);
    Py_RETURN_NONE;
}

CLINKAGE PyObject* pw_amend_insert_realwf_gpu(PyObject* self, PyObject* args)
{
    PyObject *array_nQ_obj;
    int n;
    int m;
    if (!PyArg_ParseTuple(args, "Oii",
                          &array_nQ_obj, &n, &m))
        return NULL;
    void *array_nQ = gpaw::Array_DATA<void>(array_nQ_obj);
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

    pw_amend_insert_realwf_gpu_launch_kernel(dtypenum, nb, nx, ny, nz, n, m, array_nQ);
    Py_RETURN_NONE;
}



CLINKAGE PyObject* add_to_density_gpu(PyObject* self, PyObject* args)
{
    PyObject *f_n_obj, *psit_nR_obj, *rho_R_obj;
    if (!PyArg_ParseTuple(args, "OOO",
                          &f_n_obj, &psit_nR_obj, &rho_R_obj))
        return NULL;
    int dtypenum = get_dtype(psit_nR_obj);

    double *f_n = gpaw::Array_DATA<double>(f_n_obj);
    void *psit_nR = gpaw::Array_DATA<void>(psit_nR_obj);
    void *rho_R = gpaw::Array_DATA<void>(rho_R_obj);
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
    add_to_density_gpu_launch_kernel(nb, nR, f_n, psit_nR, rho_R, dtypenum);
    Py_RETURN_NONE;
}


CLINKAGE PyObject* calculate_residual_gpu(PyObject* self, PyObject* args)
{
    PyObject *residual_nG_obj, *eps_n_obj, *wf_nG_obj;
    if (!PyArg_ParseTuple(args, "OOO",
                          &residual_nG_obj, &eps_n_obj, &wf_nG_obj))
        return NULL;
    void *residual_nG = gpaw::Array_DATA<void>(residual_nG_obj);
    void* eps_n = gpaw::Array_DATA<void>(eps_n_obj);
    void *wf_nG = gpaw::Array_DATA<void>(wf_nG_obj);
    int nn = gpaw::Array_DIM(residual_nG_obj, 0);
    int nG = 1;
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
    calculate_residual_launch_kernel(dtypenum, nG, nn, residual_nG, eps_n, wf_nG);
    Py_RETURN_NONE;
}
