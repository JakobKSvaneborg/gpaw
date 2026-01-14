/*  Copyright (C) 2003-2007  CAMP
 *  Copyright (C) 2007-2008  CAMd
 *  Copyright (C) 2008-2010  CSC - IT Center for Science Ltd.
 *  Copyright (C) 2011  Argonne National Laboratory
 *  Please see the accompanying LICENSE file for further information. */

#include "python_utils.h"
#include "extensions.h"
#include <math.h>
#include <stdlib.h>
#ifdef __DARWIN_UNIX03
/* Allows for special MaxOS magic */
#include <malloc/malloc.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif



PyObject* get_num_threads(PyObject *self, PyObject *args)
{
  int nthreads = 1;
#ifdef _OPENMP
  #pragma omp parallel
  {
    nthreads = omp_get_num_threads();
  }
#endif
  return Py_BuildValue("i", nthreads);
}

#ifdef PARALLEL
#include <mpi.h>
#endif


// returns the distance between two 3d double vectors
double distance(double *a, double *b)
{
  double sum = 0;
  double diff;
  for (int c = 0; c < 3; c++) {
    diff = a[c] - b[c];
    sum += diff*diff;
  }
  return sqrt(sum);
}

PyObject* add_to_density_gpu(PyObject* self, PyObject* args);
// Equivalent to:
//
//     nt_R += f * abs(psit_R)**2
//
PyObject* add_to_density(PyObject *self, PyObject *args)
{
    double f;
    PyArrayObject* psit_R_obj;
    PyArrayObject* nt_R_obj;
    if (!PyArg_ParseTuple(args, "dOO", &f, &psit_R_obj, &nt_R_obj))
        return NULL;

    if (PyArray_Check(psit_R_obj))
    {
        const double* psit_R = (double*) PyArray_DATA(psit_R_obj);
        double* nt_R = (double*) PyArray_DATA(nt_R_obj);
        int n = PyArray_SIZE(nt_R_obj);
        if (PyArray_ITEMSIZE(psit_R_obj) == 8) {
            // Real wave functions
            // psit_R can be a view of a larger array (psit_R = tmp[:, :, :dim2])
            int stride2 = PyArray_STRIDE(psit_R_obj, 1) / 8;
            int dim2 = PyArray_DIM(psit_R_obj, 2);
            int j = 0;
            for (int i = 0; i < n;) {
                for (int k = 0; k < dim2; i++, j++, k++)
                    nt_R[i] += f * psit_R[j] * psit_R[j];
                j += stride2 - dim2;
            }
        }
        else
            // Complex wave functions
            for (int i = 0; i < n; i++)
                nt_R[i] += f * (psit_R[2 * i] * psit_R[2 * i] +
                                psit_R[2 * i + 1] * psit_R[2 * i + 1]);
    }
    else
    {
        // Must be cupy
        #ifdef GPAW_GPU
        add_to_density_gpu(self, args);
        #else
        PyErr_SetString(PyExc_RuntimeError, "Unknown array type to add_to_density.");
        return NULL;
        #endif
    }
    Py_RETURN_NONE;
}


PyObject* utilities_gaussian_wave(PyObject *self, PyObject *args)
{
  Py_complex A_obj;
  PyArrayObject* r_cG_obj;
  PyArrayObject* r0_c_obj;
  Py_complex sigma_obj; // imaginary part ignored
  PyArrayObject* k_c_obj;
  PyArrayObject* gs_G_obj;

  if (!PyArg_ParseTuple(args, "DOODOO", &A_obj, &r_cG_obj, &r0_c_obj, &sigma_obj, &k_c_obj, &gs_G_obj))
    return NULL;

  int C, G;
  C = PyArray_DIMS(r_cG_obj)[0];
  G = PyArray_DIMS(r_cG_obj)[1];
  for (int i = 2; i < PyArray_NDIM(r_cG_obj); i++)
        G *= PyArray_DIMS(r_cG_obj)[i];

  double* r_cG = DOUBLEP(r_cG_obj); // XXX not ideally strided
  double* r0_c = DOUBLEP(r0_c_obj);
  double dr2, kr, alpha = -0.5/pow(sigma_obj.real, 2);

  int gammapoint = 1;
  double* k_c = DOUBLEP(k_c_obj);
  for (int c=0; c<C; c++)
    gammapoint &= (k_c[c]==0);

  if (PyArray_DESCR(gs_G_obj)->type_num == NPY_DOUBLE)
    {
      double* gs_G = DOUBLEP(gs_G_obj);

      if(gammapoint)
        for(int g=0; g<G; g++)
          {
            dr2 = pow(r_cG[g]-r0_c[0], 2);
            for(int c=1; c<C; c++)
              dr2 += pow(r_cG[c*G+g]-r0_c[c], 2);
            gs_G[g] = A_obj.real*exp(alpha*dr2);
          }
      else if(sigma_obj.real>0)
        for(int g=0; g<G; g++)
          {
            kr = k_c[0]*r_cG[g];
            dr2 = pow(r_cG[g]-r0_c[0], 2);
            for(int c=1; c<C; c++)
              {
                kr += k_c[c]*r_cG[c*G+g];
                dr2 += pow(r_cG[c*G+g]-r0_c[c], 2);
              }
            kr = A_obj.real*cos(kr)-A_obj.imag*sin(kr);
            gs_G[g] = kr*exp(alpha*dr2);
          }
      else
        {
          return NULL; //TODO sigma<=0 could be exp(-(r-r0)^2/2sigma^2) -> 1 ?
        }
    }
  else
    {
      double_complex* gs_G = COMPLEXP(gs_G_obj);
      double_complex A = A_obj.real+I*A_obj.imag;

      if(gammapoint)
        for(int g=0; g<G; g++)
          {
            dr2 = pow(r_cG[g]-r0_c[0], 2);
            for(int c=1; c<C; c++)
              dr2 += pow(r_cG[c*G+g]-r0_c[c], 2);
            gs_G[g] = A*exp(alpha*dr2);
          }
      else if(sigma_obj.real>0)
        for(int g=0; g<G; g++)
          {
            kr = k_c[0]*r_cG[g];
            dr2 = pow(r_cG[g]-r0_c[0], 2);
            for(int c=1; c<C; c++)
              {
                kr += k_c[c]*r_cG[c*G+g];
                dr2 += pow(r_cG[c*G+g]-r0_c[c], 2);
              }
            double_complex f = A*cexp(I*kr);
            gs_G[g] = f*exp(alpha*dr2);
          }
      else
        {
          return NULL; //TODO sigma<=0 could be exp(-(r-r0)^2/2sigma^2) -> 1 ?
        }
    }

  Py_RETURN_NONE;
}


PyObject* pack(PyObject *self, PyObject *args)
{
    PyArrayObject* a_obj;
    if (!PyArg_ParseTuple(args, "O", &a_obj))
        return NULL;
    a_obj = PyArray_GETCONTIGUOUS(a_obj);
    int n = PyArray_DIMS(a_obj)[0];
    npy_intp dims[1] = {n * (n + 1) / 2};
    int typenum = PyArray_DESCR(a_obj)->type_num;
    PyArrayObject* b_obj = (PyArrayObject*) PyArray_SimpleNew(1, dims,
                                                              typenum);
    if (b_obj == NULL)
      return NULL;
    if (typenum == NPY_DOUBLE) {
        double* a = (double*)PyArray_DATA(a_obj);
        double* b = (double*)PyArray_DATA(b_obj);
        for (int r = 0; r < n; r++) {
            *b++ = a[r + n * r];
            for (int c = r + 1; c < n; c++)
                *b++ = a[r + n * c] + a[c + n * r];
        }
    } else {
        double_complex* a = (double_complex*)PyArray_DATA(a_obj);
        double_complex* b = (double_complex*)PyArray_DATA(b_obj);
        for (int r = 0; r < n; r++) {
            *b++ = a[r + n * r];
            for (int c = r + 1; c < n; c++)
                *b++ = a[r + n * c] + a[c + n * r];
        }
    }
    Py_DECREF(a_obj);
    PyObject* value = Py_BuildValue("O", b_obj);
    Py_DECREF(b_obj);
    return value;
}


PyObject* unpack(PyObject *self, PyObject *args)
{
  PyArrayObject* ap;
  PyArrayObject* a;
  if (!PyArg_ParseTuple(args, "OO", &ap, &a))
    return NULL;
  int n = PyArray_DIMS(a)[0];
  double* datap = DOUBLEP(ap);
  double* data = DOUBLEP(a);
  for (int r = 0; r < n; r++)
    for (int c = r; c < n; c++)
      {
        double d = *datap++;
        data[c + r * n] = d;
        data[r + c * n] = d;
      }
  Py_RETURN_NONE;
}

PyObject* unpack_complex(PyObject *self, PyObject *args)
{
  PyArrayObject* ap;
  PyArrayObject* a;
  if (!PyArg_ParseTuple(args, "OO", &ap, &a))
    return NULL;
  int n = PyArray_DIMS(a)[0];
  double_complex* datap = COMPLEXP(ap);
  double_complex* data = COMPLEXP(a);
  for (int r = 0; r < n; r++)
    for (int c = r; c < n; c++)
      {
        double_complex d = *datap++;
        data[c + r * n] = d;
        data[r + c * n] = conj(d);
      }
  Py_RETURN_NONE;
}

PyObject* hartree(PyObject *self, PyObject *args)
{
    int l;
    PyArrayObject* nrdr_obj;
    PyArrayObject* r_obj;
    PyArrayObject* vr_obj;
    if (!PyArg_ParseTuple(args, "iOOO", &l, &nrdr_obj, &r_obj, &vr_obj))
        return NULL;

    const int M = PyArray_DIM(nrdr_obj, 0);
    const double* nrdr = DOUBLEP(nrdr_obj);
    const double* r = DOUBLEP(r_obj);
    double* vr = DOUBLEP(vr_obj);

    double p = 0.0;
    double q = 0.0;
    for (int g = M - 1; g > 0; g--)
    {
        double R = r[g];
        double rl = pow(R, l);
        double dp = nrdr[g] / rl;
        double rlp1 = rl * R;
        double dq = nrdr[g] * rlp1;
        vr[g] = (p + 0.5 * dp) * rlp1 - (q + 0.5 * dq) / rl;
        p += dp;
        q += dq;
    }
    vr[0] = 0.0;
    double f = 4.0 * M_PI / (2 * l + 1);
    for (int g = 1; g < M; g++)
    {
        double R = r[g];
        vr[g] = f * (vr[g] + q / pow(R, l));
    }
    Py_RETURN_NONE;
}

PyObject* localize(PyObject *self, PyObject *args)
{
  PyArrayObject* Z_nnc;
  PyArrayObject* U_nn;
  if (!PyArg_ParseTuple(args, "OO", &Z_nnc, &U_nn))
    return NULL;

  const int n = PyArray_DIMS(U_nn)[0];

  // The original C99 implementation of this function made heavy use of VLA
  // semantics that did not compile on Clang++ in C++ mode (but g++ was OK).
  // Here we attempt to minimize code changes by using custom accessors to index multidimensional arrays.
  // If compiling as C code, the accessors fall back to usual VLA syntax.

#if GPAW_CPP
  double_complex* Z = (double_complex*)PyArray_DATA(Z_nnc);
  double* U = (double*)PyArray_DATA(U_nn);

  auto Z_at = [&](int i, int j, int c) -> double_complex&
  {
      return Z[(i * n + j) * 3 + c];
  };

  auto U_at = [&](int i, int j) -> double&
  {
      return U[i * n + j];
  };

#else
  // Old C99 version but using similar accessor helper as the C++ version
  double_complex (*Z)[n][3] = (double_complex (*)[n][3])COMPLEXP(Z_nnc);
  double (*U)[n] = (double (*)[n])DOUBLEP(U_nn);
  #define Z_at(i, j, c) Z[i][j][c]
  #define U_at(i, j)    U[i][j]
#endif

  double value = 0.0;
  for (int a = 0; a < n; a++)
    {
      for (int b = a + 1; b < n; b++)
        {
          double x = 0.0;
          double y = 0.0;
          for (int c = 0; c < 3; c++)
            {
              const double_complex Zaac = Z_at(a, a, c);
              const double_complex Zabc = Z_at(a, b, c);
              const double_complex Zbbc = Z_at(b, b, c);

              x += (0.25 * creal(Zbbc * conj(Zbbc)) +
                    0.25 * creal(Zaac * conj(Zaac)) -
                    0.5 * creal(Zaac * conj(Zbbc)) -
                    creal(Zabc * conj(Zabc)));
              y += creal((Zaac - Zbbc) * conj(Zabc));
            }
          double t = 0.25 * atan2(y, x);
          double C = cos(t);
          double S = sin(t);
          for (int i = 0; i < a; i++)
            for (int c = 0; c < 3; c++)
              {
                double_complex Ziac = Z_at(i, a, c);
                Z_at(i, a, c) = C * Ziac + S * Z_at(i, b, c);
                Z_at(i, b, c) = C * Z_at(i, b, c) - S * Ziac;
              }
          for (int c = 0; c < 3; c++)
            {
              const double_complex Zaac = Z_at(a, a, c);
              const double_complex Zabc = Z_at(a, b, c);
              const double_complex Zbbc = Z_at(b, b, c);
              Z_at(a, a, c) = C * C * Zaac + 2 * C * S * Zabc + S * S * Zbbc;
              Z_at(b, b, c) = C * C * Zbbc - 2 * C * S * Zabc + S * S * Zaac;
              Z_at(a, b, c) = S * C * (Zbbc - Zaac) + (C * C - S * S) * Zabc;
            }
          for (int i = a + 1; i < b; i++)
            for (int c = 0; c < 3; c++)
              {
                const double_complex Zaic = Z_at(a, i, c);
                Z_at(a, i, c) = C * Zaic + S * Z_at(i, b, c);
                Z_at(i, b, c) = C * Z_at(i, b, c) - S * Zaic;
              }
          for (int i = b + 1; i < n; i++)
            for (int c = 0; c < 3; c++)
              {
                const double_complex Zaic = Z_at(a, i, c);
                Z_at(a, i, c) = C * Zaic + S * Z_at(b, i, c);
                Z_at(b, i, c) = C * Z_at(b, i, c) - S * Zaic;
              }
          for (int i = 0; i < n; i++)
            {
              const double Uia = U_at(i, a);
              U_at(i, a) = C * Uia + S * U_at(i, b);
              U_at(i, b) = C * U_at(i, b) - S * Uia;
            }
        }

      for (int c = 0; c < 3; c++)
      {
        const double_complex Zaac = Z_at(a, a, c);
        value += creal(Zaac * conj(Zaac));
      }
    }
  return Py_BuildValue("d", value);
}


PyObject* spherical_harmonics(PyObject *self, PyObject *args)
{
  int l;
  PyArrayObject* R_obj_c;
  PyArrayObject* Y_obj_m;
  if (!PyArg_ParseTuple(args, "iOO", &l, &R_obj_c, &Y_obj_m))
    return NULL;

  double* R_c = DOUBLEP(R_obj_c);
  double* Y_m = DOUBLEP(Y_obj_m);

  if (l == 0)
      Y_m[0] = 0.28209479177387814;
  else
    {
      double x = R_c[0];
      double y = R_c[1];
      double z = R_c[2];
      if (l == 1)
        {
          Y_m[0] = 0.48860251190291992 * y;
          Y_m[1] = 0.48860251190291992 * z;
          Y_m[2] = 0.48860251190291992 * x;
        }
      else
        {
          double r2 = x*x+y*y+z*z;
          if (l == 2)
            {
              Y_m[0] = 1.0925484305920792 * x*y;
              Y_m[1] = 1.0925484305920792 * y*z;
              Y_m[2] = 0.31539156525252005 * (3*z*z-r2);
              Y_m[3] = 1.0925484305920792 * x*z;
              Y_m[4] = 0.54627421529603959 * (x*x-y*y);
            }
          else if (l == 3)
            {
              Y_m[0] = 0.59004358992664352 * (-y*y*y+3*x*x*y);
              Y_m[1] = 2.8906114426405538 * x*y*z;
              Y_m[2] = 0.45704579946446577 * (-y*r2+5*y*z*z);
              Y_m[3] = 0.3731763325901154 * (5*z*z*z-3*z*r2);
              Y_m[4] = 0.45704579946446577 * (5*x*z*z-x*r2);
              Y_m[5] = 1.4453057213202769 * (x*x*z-y*y*z);
              Y_m[6] = 0.59004358992664352 * (x*x*x-3*x*y*y);
            }
          else if (l == 4)
            {
              Y_m[0] = 2.5033429417967046 * (x*x*x*y-x*y*y*y);
              Y_m[1] = 1.7701307697799307 * (-y*y*y*z+3*x*x*y*z);
              Y_m[2] = 0.94617469575756008 * (-x*y*r2+7*x*y*z*z);
              Y_m[3] = 0.66904654355728921 * (-3*y*z*r2+7*y*z*z*z);
              Y_m[4] = 0.10578554691520431 * (-30*z*z*r2+3*r2*r2+35*z*z*z*z);
              Y_m[5] = 0.66904654355728921 * (7*x*z*z*z-3*x*z*r2);
              Y_m[6] = 0.47308734787878004 * (-x*x*r2+7*x*x*z*z+y*y*r2-7*y*y*z*z);
              Y_m[7] = 1.7701307697799307 * (x*x*x*z-3*x*y*y*z);
              Y_m[8] = 0.62583573544917614 * (-6*x*x*y*y+x*x*x*x+y*y*y*y);
            }
          else if (l == 5)
            {
              Y_m[0] = 0.65638205684017015 * (y*y*y*y*y+5*x*x*x*x*y-10*x*x*y*y*y);
              Y_m[1] = 8.3026492595241645 * (x*x*x*y*z-x*y*y*y*z);
              Y_m[2] = 0.48923829943525038 * (y*y*y*r2-9*y*y*y*z*z-3*x*x*y*r2+27*x*x*y*z*z);
              Y_m[3] = 4.7935367849733241 * (3*x*y*z*z*z-x*y*z*r2);
              Y_m[4] = 0.45294665119569694 * (-14*y*z*z*r2+y*r2*r2+21*y*z*z*z*z);
              Y_m[5] = 0.1169503224534236 * (63*z*z*z*z*z+15*z*r2*r2-70*z*z*z*r2);
              Y_m[6] = 0.45294665119569694 * (x*r2*r2-14*x*z*z*r2+21*x*z*z*z*z);
              Y_m[7] = 2.3967683924866621 * (-3*y*y*z*z*z+y*y*z*r2+3*x*x*z*z*z-x*x*z*r2);
              Y_m[8] = 0.48923829943525038 * (9*x*x*x*z*z-27*x*y*y*z*z-x*x*x*r2+3*x*y*y*r2);
              Y_m[9] = 2.0756623148810411 * (y*y*y*y*z-6*x*x*y*y*z+x*x*x*x*z);
              Y_m[10] = 0.65638205684017015 * (-10*x*x*x*y*y+5*x*y*y*y*y+x*x*x*x*x);
            }
          else if (l == 6)
            {
              Y_m[0] = 1.3663682103838286 * (-10*x*x*x*y*y*y+3*x*x*x*x*x*y+3*x*y*y*y*y*y);
              Y_m[1] = 2.3666191622317521 * (y*y*y*y*y*z-10*x*x*y*y*y*z+5*x*x*x*x*y*z);
              Y_m[2] = 2.0182596029148967 * (-x*x*x*y*r2+x*y*y*y*r2-11*x*y*y*y*z*z+11*x*x*x*y*z*z);
              Y_m[3] = 0.92120525951492349 * (-11*y*y*y*z*z*z-9*x*x*y*z*r2+33*x*x*y*z*z*z+3*y*y*y*z*r2);
              Y_m[4] =0.92120525951492349 * (x*y*r2*r2+33*x*y*z*z*z*z-18*x*y*z*z*r2);
              Y_m[5] = 0.58262136251873142 * (5*y*z*r2*r2+33*y*z*z*z*z*z-30*y*z*z*z*r2);
              Y_m[6] = 0.063569202267628425 * (231*z*z*z*z*z*z-5*r2*r2*r2+105*z*z*r2*r2-315*z*z*z*z*r2);
              Y_m[7] = 0.58262136251873142 * (-30*x*z*z*z*r2+33*x*z*z*z*z*z+5*x*z*r2*r2);
              Y_m[8] = 0.46060262975746175 * (33*x*x*z*z*z*z+x*x*r2*r2-y*y*r2*r2-18*x*x*z*z*r2+18*y*y*z*z*r2-33*y*y*z*z*z*z);
              Y_m[9] = 0.92120525951492349 * (-3*x*x*x*z*r2-33*x*y*y*z*z*z+9*x*y*y*z*r2+11*x*x*x*z*z*z);
              Y_m[10] = 0.50456490072872417 * (11*y*y*y*y*z*z-66*x*x*y*y*z*z-x*x*x*x*r2+6*x*x*y*y*r2+11*x*x*x*x*z*z-y*y*y*y*r2);
              Y_m[11] = 2.3666191622317521 * (5*x*y*y*y*y*z+x*x*x*x*x*z-10*x*x*x*y*y*z);
              Y_m[12] = 0.6831841051919143 * (x*x*x*x*x*x+15*x*x*y*y*y*y-15*x*x*x*x*y*y-y*y*y*y*y*y);
            }
          else if (l == 7)
            {
              Y_m[0] = -0.707162732524596 * y*y*y*y*y*y*y + 14.8504173830165 * x*x*y*y*y*y*y + -24.7506956383609 * x*x*x*x*y*y*y + 4.95013912767217 * x*x*x*x*x*x*y;
              Y_m[1] = 15.8757639708114 * x*y*y*y*y*y*z + -52.919213236038 * x*x*x*y*y*y*z + 15.8757639708114 * x*x*x*x*x*y*z;
              Y_m[2] = 4.67024020848234 * x*x*y*y*y*y*y + -0.51891557872026 * y*y*y*y*y*y*y + 6.22698694464312 * y*y*y*y*y*z*z + 2.5945778936013 * x*x*x*x*y*y*y + -62.2698694464312 * x*x*y*y*y*z*z + -2.5945778936013 * x*x*x*x*x*x*y + 31.1349347232156 * x*x*x*x*y*z*z;
              Y_m[3] = 12.4539738892862 * x*y*y*y*y*y*z + -41.5132462976208 * x*y*y*y*z*z*z + -12.4539738892862 * x*x*x*x*x*y*z + 41.5132462976208 * x*x*x*y*z*z*z;
              Y_m[4] = 2.34688400793441 * x*x*x*x*y*y*y + 0.469376801586882 * x*x*y*y*y*y*y + -18.7750720634753 * x*x*y*y*y*z*z + -0.469376801586882 * y*y*y*y*y*y*y + 9.38753603173764 * y*y*y*y*y*z*z + -12.5167147089835 * y*y*y*z*z*z*z + 1.40813040476065 * x*x*x*x*x*x*y + -28.1626080952129 * x*x*x*x*y*z*z + 37.5501441269506 * x*x*y*z*z*z*z;
              Y_m[5] = 6.63799038667474 * x*x*x*x*x*y*z + 13.2759807733495 * x*x*x*y*y*y*z + -35.4026153955986 * x*x*x*y*z*z*z + 6.63799038667474 * x*y*y*y*y*y*z + -35.4026153955986 * x*y*y*y*z*z*z + 21.2415692373592 * x*y*z*z*z*z*z;
              Y_m[6] = -0.451658037912587 * x*x*x*x*x*x*y + -1.35497411373776 * x*x*x*x*y*y*y + 10.8397929099021 * x*x*x*x*y*z*z + -1.35497411373776 * x*x*y*y*y*y*y + 21.6795858198042 * x*x*y*y*y*z*z + -21.6795858198042 * x*x*y*z*z*z*z + -0.451658037912587 * y*y*y*y*y*y*y + 10.8397929099021 * y*y*y*y*y*z*z + -21.6795858198042 * y*y*y*z*z*z*z + 5.78122288528111 * y*z*z*z*z*z*z;
              Y_m[7] = -2.38994969192017 * x*x*x*x*x*x*z + -7.16984907576052 * x*x*x*x*y*y*z + 14.339698151521 * x*x*x*x*z*z*z + -7.16984907576052 * x*x*y*y*y*y*z + 28.6793963030421 * x*x*y*y*z*z*z + -11.4717585212168 * x*x*z*z*z*z*z + -2.38994969192017 * y*y*y*y*y*y*z + 14.339698151521 * y*y*y*y*z*z*z + -11.4717585212168 * y*y*z*z*z*z*z + 1.09254843059208 * z*z*z*z*z*z*z;
              Y_m[8] = -0.451658037912587 * x*x*x*x*x*x*x + -1.35497411373776 * x*x*x*x*x*y*y + 10.8397929099021 * x*x*x*x*x*z*z + -1.35497411373776 * x*x*x*y*y*y*y + 21.6795858198042 * x*x*x*y*y*z*z + -21.6795858198042 * x*x*x*z*z*z*z + -0.451658037912587 * x*y*y*y*y*y*y + 10.8397929099021 * x*y*y*y*y*z*z + -21.6795858198042 * x*y*y*z*z*z*z + 5.78122288528111 * x*z*z*z*z*z*z;
              Y_m[9] = -3.31899519333737 * x*x*y*y*y*y*z + -3.31899519333737 * y*y*y*y*y*y*z + 17.7013076977993 * y*y*y*y*z*z*z + -10.6207846186796 * y*y*z*z*z*z*z + 3.31899519333737 * x*x*x*x*x*x*z + 3.31899519333737 * x*x*x*x*y*y*z + -17.7013076977993 * x*x*x*x*z*z*z + 10.6207846186796 * x*x*z*z*z*z*z;
              Y_m[10] = -0.469376801586882 * x*x*x*x*x*y*y + -2.34688400793441 * x*x*x*y*y*y*y + 18.7750720634753 * x*x*x*y*y*z*z + -1.40813040476065 * x*y*y*y*y*y*y + 28.1626080952129 * x*y*y*y*y*z*z + -37.5501441269506 * x*y*y*z*z*z*z + 0.469376801586882 * x*x*x*x*x*x*x + -9.38753603173764 * x*x*x*x*x*z*z + 12.5167147089835 * x*x*x*z*z*z*z;
              Y_m[11] = 15.5674673616078 * x*x*y*y*y*y*z + -3.11349347232156 * y*y*y*y*y*y*z + 10.3783115744052 * y*y*y*y*z*z*z + 15.5674673616078 * x*x*x*x*y*y*z + -62.2698694464312 * x*x*y*y*z*z*z + -3.11349347232156 * x*x*x*x*x*x*z + 10.3783115744052 * x*x*x*x*z*z*z;
              Y_m[12] = 2.5945778936013 * x*x*x*y*y*y*y + -2.5945778936013 * x*y*y*y*y*y*y + 31.1349347232156 * x*y*y*y*y*z*z + 4.67024020848234 * x*x*x*x*x*y*y + -62.2698694464312 * x*x*x*y*y*z*z + -0.51891557872026 * x*x*x*x*x*x*x + 6.22698694464312 * x*x*x*x*x*z*z;
              Y_m[13] = -2.6459606618019 * y*y*y*y*y*y*z + 39.6894099270285 * x*x*y*y*y*y*z + -39.6894099270285 * x*x*x*x*y*y*z + 2.6459606618019 * x*x*x*x*x*x*z;
              Y_m[14] = -4.95013912767217 * x*y*y*y*y*y*y + 24.7506956383609 * x*x*x*y*y*y*y + -14.8504173830165 * x*x*x*x*x*y*y + 0.707162732524596 * x*x*x*x*x*x*x;
            }
          else
            {
              PyErr_SetString(PyExc_RuntimeError, "l>7 not implemented");
              return NULL;
            }
        }
    }
  Py_RETURN_NONE;
}


PyObject* integrate_outwards(PyObject *self, PyObject *args)
{
    int g0;
    PyArrayObject* cm1_g_obj;
    PyArrayObject* c0_g_obj;
    PyArrayObject* cp1_g_obj;
    PyArrayObject* b_g_obj;
    PyArrayObject* a_g_obj;

    if (!PyArg_ParseTuple(args,
                          "iOOOOO",
                          &g0, &cm1_g_obj, &c0_g_obj, &cp1_g_obj,
                          &b_g_obj, &a_g_obj))
        return NULL;

    const double* cm1_g = DOUBLEP(cm1_g_obj);
    const double* c0_g = DOUBLEP(c0_g_obj);
    const double* cp1_g = DOUBLEP(cp1_g_obj);
    const double* b_g = DOUBLEP(b_g_obj);
    double* a_g = DOUBLEP(a_g_obj);

    for (int g = 1; g <= g0; g++)
        a_g[g + 1] = -(a_g[g - 1] * cm1_g[g] +
                       a_g[g] * c0_g[g] + b_g[g]) / cp1_g[g];

    Py_RETURN_NONE;
}


PyObject* integrate_inwards(PyObject *self, PyObject *args)
{
    int g1, g0;
    PyArrayObject* c0_g_obj;
    PyArrayObject* cp1_g_obj;
    PyArrayObject* a_g_obj;

    if (!PyArg_ParseTuple(args,
                          "iiOOO",
                          &g1, &g0, &c0_g_obj, &cp1_g_obj, &a_g_obj))
        return NULL;

    const double* c0_g = DOUBLEP(c0_g_obj);
    const double* cp1_g = DOUBLEP(cp1_g_obj);
    double* a_g = DOUBLEP(a_g_obj);
    const int ng = PyArray_DIM(a_g_obj, 0);

    for (int g = g1; g >= g0; g--) {
        double ag = a_g[g];
        if (ag > 1e50) {
            for (int gg = g; gg < ng; gg++)
                a_g[gg] = a_g[gg] / 1e50;
            ag = ag / 1e50;
        }
        a_g[g - 1] = a_g[g + 1] * cp1_g[g] + ag * c0_g[g];
    }
    Py_RETURN_NONE;
}
