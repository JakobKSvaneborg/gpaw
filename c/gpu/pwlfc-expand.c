#include "../extensions.h"

#define GPAW_ARRAY_DISABLE_NUMPY
#define GPAW_ARRAY_ALLOW_CUPY
#include "../array.h"
#undef GPAW_ARRAY_DISABLE_NUMPY

#include "gpu.h"
#include "gpu-complex.h"
#include <stdio.h>

void calculate_residual_launch_kernel(int nG,
                                      int nn,
                                      double* residual_ng, 
                                      double* eps_n, 
                                      double* wf_nG,
                                      int is_complex);

void pwlfc_expand_gpu_launch_kernel(int itemsize,
                                    double* f_Gs,
                                    gpuDoubleComplex *emiGR_Ga,
                                    double *Y_GL,
                                    int* l_s,
                                    int* a_J,
                                    int* s_J,
                                    double* f_GI,
                                    int* I_J,
                                    int nG,
                                    int nJ,
                                    int nL,
                                    int nI,
                                    int natoms,
                                    int nsplines,
                                    bool cc);

void pw_insert_gpu_launch_kernel(
                             int nb,
                             int nG,
                             int nQ,
                             double* c_nG,
                             npy_int32* Q_G,
                             double scale,
                             double* tmp_nQ);

void add_to_density_gpu_launch_kernel(int nb,
                                      int nR,
                                      double* f_n,
                                      double complex* psit_nR,
                                      double* rho_R,
                                      int wfs_is_complex);


void dH_aii_times_P_ani_launch_kernel(int nA, int nn,
                                      int nI, npy_int32* ni_a, 
                                      double* dH_aii_dev, 
                                      gpuDoubleComplex* P_ani_dev,
                                      gpuDoubleComplex* outP_ani_dev,
                                      int is_complex);

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

PyObject* evaluate_lda_gpu(PyObject* self, PyObject* args)
{
    PyObject* n_obj;
    PyObject* v_obj;
    PyObject* e_obj; 
    if (!PyArg_ParseTuple(args, "OOO",
                          &n_obj, &v_obj, &e_obj))
        return NULL;
    int nspin = Array_DIM(n_obj, 0);
    if ((nspin != 1) && (nspin != 2))
    {
        PyErr_Format(PyExc_RuntimeError, "Expected 1 or 2 spins. Got %d.", nspin);
        return NULL;
    }
    int ng = 1;
    for (int d=1; d<Array_NDIM(n_obj); d++)
    {
        ng *= Array_DIM(n_obj, d);
    }
    double* n_ptr = Array_DATA(n_obj);
    double* v_ptr = Array_DATA(v_obj);
    double* e_ptr = Array_DATA(e_obj);
    if (PyErr_Occurred())
    {
        return NULL;
    }
    evaluate_lda_launch_kernel(nspin, ng,
                               n_ptr, v_ptr, e_ptr);
    Py_RETURN_NONE;
}

PyObject* evaluate_pbe_gpu(PyObject* self, PyObject* args)
{
    PyObject* n_obj;
    PyObject* v_obj;
    PyObject* sigma_obj;
    PyObject* dedsigma_obj;
    PyObject* e_obj; 
    if (!PyArg_ParseTuple(args, "OOOOO",
                          &n_obj, &v_obj, &e_obj, &sigma_obj, &dedsigma_obj))
        return NULL;
    int nspin = Array_DIM(n_obj, 0);
    if ((nspin != 1) && (nspin != 2))
    {
        PyErr_Format(PyExc_RuntimeError, "Expected 1 or 2 spins. Got %d.", nspin);
        return NULL;
    }
    int ng = 1;
    for (int d=1; d<Array_NDIM(n_obj); d++)
    {
        ng *= Array_DIM(n_obj, d);
    }
    double* n_ptr = Array_DATA(n_obj);
    double* v_ptr = Array_DATA(v_obj);
    double* e_ptr = Array_DATA(e_obj);
    double* sigma_ptr = Array_DATA(sigma_obj);
    double* dedsigma_ptr = Array_DATA(dedsigma_obj);
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

PyObject* dH_aii_times_P_ani_gpu(PyObject* self, PyObject* args)
{
    PyObject* dH_aii_obj;
    PyObject* ni_a_obj;
    PyObject* P_ani_obj;
    PyObject* outP_ani_obj;
    if (!PyArg_ParseTuple(args, "OOOO",
                          &dH_aii_obj, &ni_a_obj, &P_ani_obj, &outP_ani_obj))
        return NULL;


    if (Array_DIM(ni_a_obj, 0) == 0)
    {
        Py_RETURN_NONE;
    }

    double* dH_aii_dev = Array_DATA(dH_aii_obj);
    if (!dH_aii_dev) 
    {
	PyErr_SetString(PyExc_RuntimeError, "Error in input dH_aii.");
        return NULL;
    }
    gpuDoubleComplex* P_ani_dev = Array_DATA(P_ani_obj);
    if (!P_ani_dev)
    {
        PyErr_SetString(PyExc_RuntimeError, "Error in input P_ani.");
        return NULL;
    }
    gpuDoubleComplex* outP_ani_dev = Array_DATA(outP_ani_obj);
    if (!outP_ani_dev) 
    {
        PyErr_SetString(PyExc_RuntimeError, "Error in output outP_ani.");
        return NULL;
    }
    npy_int32* ni_a = Array_DATA(ni_a_obj);
    if (!ni_a) 
    {
        PyErr_SetString(PyExc_RuntimeError, "Error in input ni_a.");
        return NULL;
    }

    int is_complex = Array_ITEMSIZE(P_ani_obj) == 16;
    if (Array_ITEMSIZE(P_ani_obj) != Array_ITEMSIZE(outP_ani_obj))
    {
        PyErr_SetString(PyExc_RuntimeError, "Incompatible P_ani and outP_ani.");
        return NULL;
    }
    assert(Array_ITEMSIZE(dH_aii_obj) == 8);
    assert(Array_ITEMSIZE(ni_a_obj) == 4);

    int nA = Array_DIM(ni_a_obj, 0);
    int nn = Array_DIM(P_ani_obj, 0);
    int nI = Array_DIM(P_ani_obj, 1);
    if (PyErr_Occurred())
    {
        return NULL;
    }

    dH_aii_times_P_ani_launch_kernel(nA, nn, nI, ni_a, dH_aii_dev, P_ani_dev, outP_ani_dev, is_complex);
    Py_RETURN_NONE;
}


PyObject* pwlfc_expand_gpu(PyObject* self, PyObject* args)
{
    PyObject *f_Gs_obj;
    PyObject *emiGR_Ga_obj;
    PyObject *Y_GL_obj;
    PyObject *l_s_obj;
    PyObject *a_J_obj;
    PyObject *s_J_obj;
    int cc;
    PyObject *f_GI_obj;
    PyObject *I_J_obj;

    if (!PyArg_ParseTuple(args, "OOOOOOiOO",
                          &f_Gs_obj, &emiGR_Ga_obj, &Y_GL_obj,
                          &l_s_obj, &a_J_obj, &s_J_obj,
                          &cc, &f_GI_obj, &I_J_obj))
        return NULL;
    double *f_Gs = (double*)Array_DATA(f_Gs_obj);
    double *Y_GL = (double*)Array_DATA(Y_GL_obj);
    int *l_s = (int*)Array_DATA(l_s_obj);
    int *a_J = (int*)Array_DATA(a_J_obj);
    int *s_J = (int*)Array_DATA(s_J_obj);
    double *f_GI = (double*)Array_DATA(f_GI_obj);
    int nG = Array_DIM(emiGR_Ga_obj, 0);
    int *I_J = (int*)Array_DATA(I_J_obj);
    int nJ = Array_DIM(a_J_obj, 0);
    int nL = Array_DIM(Y_GL_obj, 1);
    int nI = Array_DIM(f_GI_obj, 1);
    int natoms = Array_DIM(emiGR_Ga_obj, 1);
    int nsplines = Array_DIM(f_Gs_obj, 1);
    gpuDoubleComplex* emiGR_Ga = (gpuDoubleComplex*)Array_DATA(emiGR_Ga_obj);
    int itemsize = Array_ITEMSIZE(f_GI_obj);
    if (PyErr_Occurred())
    {
        return NULL;
    }
    pwlfc_expand_gpu_launch_kernel(itemsize, f_Gs, emiGR_Ga, Y_GL, l_s, a_J, s_J, f_GI,
                                   I_J, nG, nJ, nL, nI, natoms, nsplines, cc);
    Py_RETURN_NONE;
}

PyObject* pw_insert_gpu(PyObject* self, PyObject* args)
{
    PyObject *c_nG_obj, *Q_G_obj, *tmp_nQ_obj;
    double scale;
    if (!PyArg_ParseTuple(args, "OOdO",
                          &c_nG_obj, &Q_G_obj, &scale, &tmp_nQ_obj))
        return NULL;
    npy_int32 *Q_G = Array_DATA(Q_G_obj);
    double complex *c_nG = Array_DATA(c_nG_obj);
    double complex *tmp_nQ = Array_DATA(tmp_nQ_obj);
    int nG = 0;
    int nQ = 0;
    int nb = 0;
    assert(Array_NDIM(c_nG_obj) == Array_NDIM(tmp_nQ_obj));
    assert(Array_ITEMSIZE(c_nG_obj) == 16);
    assert(Array_ITEMSIZE(tmp_nQ_obj) == 16);
    if (Array_NDIM(c_nG_obj) == 1)
    {
        nG = Array_DIM(c_nG_obj, 0);
        nb = 1;
        nQ = Array_DIM(tmp_nQ_obj, 0);
    }
    else
    {
        nG = Array_DIM(c_nG_obj, 1);
        nb = Array_DIM(c_nG_obj, 0);
        nQ = Array_DIM(tmp_nQ_obj, 1);
    }
    if (PyErr_Occurred())
    {
        return NULL;
    }

    pw_insert_gpu_launch_kernel(nb, nG, nQ,
                                (double*)c_nG,
                                Q_G,
                                scale,
                                (double*)tmp_nQ);
    Py_RETURN_NONE;
}


PyObject* add_to_density_gpu(PyObject* self, PyObject* args)
{
    PyObject *f_n_obj, *psit_nR_obj, *rho_R_obj;
    if (!PyArg_ParseTuple(args, "OOO",
                          &f_n_obj, &psit_nR_obj, &rho_R_obj))
        return NULL;
    double *f_n = Array_DATA(f_n_obj);
    double complex *psit_nR = Array_DATA(psit_nR_obj);
    double* rho_R = Array_DATA(rho_R_obj);
    int nb = Array_SIZE(f_n_obj);
    int nR = Array_SIZE(psit_nR_obj) / nb;
    assert(Array_ITEMSIZE(rho_R_obj) == 8);
    if (PyErr_Occurred())
    {
        return NULL;
    }
    add_to_density_gpu_launch_kernel(nb, nR, f_n, psit_nR, rho_R, Array_ITEMSIZE(psit_nR_obj)==16); 
    Py_RETURN_NONE;
}


PyObject* calculate_residual_gpu(PyObject* self, PyObject* args)
{
    PyObject *residual_nG_obj, *eps_n_obj, *wf_nG_obj;
    if (!PyArg_ParseTuple(args, "OOO",
                          &residual_nG_obj, &eps_n_obj, &wf_nG_obj))
        return NULL;
    double *residual_nG = Array_DATA(residual_nG_obj);
    double* eps_n = Array_DATA(eps_n_obj);
    double *wf_nG = Array_DATA(wf_nG_obj);
    int nn = Array_DIM(residual_nG_obj, 0);
    bool is_complex = Array_ITEMSIZE(residual_nG_obj) == 16;
    int nG = 1;
    for (int d=1; d<Array_NDIM(residual_nG_obj); d++)
    {
        nG *= Array_DIM(residual_nG_obj, d);
    }
    if (PyErr_Occurred())
    {
        return NULL;
    }
    calculate_residual_launch_kernel(nG, nn, residual_nG, eps_n, wf_nG, is_complex);
    Py_RETURN_NONE;
}
