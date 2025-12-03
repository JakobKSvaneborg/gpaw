#ifdef GPAW_WITH_FFTW
#include "python_utils.h"
#include <fftw3.h>

#include "gpaw_complex.h"

#define assert_compatible_dtype(arr, ctype) \
    assert(PyArray_ITEMSIZE(arr) == sizeof(ctype));


/* Create plan and return the plan handle as Python integer */
PyObject * FFTWPlan(PyObject *self, PyObject *args)
{
    PyObject* in_obj;
    PyObject* out_obj;
    int sign;
    unsigned int flags;
    if (!PyArg_ParseTuple(args, "OOiI", &in_obj, &out_obj, &sign, &flags))
    {
        return NULL;
    }

    if (!PyArray_Check(in_obj) || !PyArray_Check(out_obj))
    {
        PyErr_SetString(PyExc_ValueError, "Invalid input: not an array");
        return NULL;
    }

    PyArrayObject* in = (PyArrayObject*)in_obj;
    PyArrayObject* out = (PyArrayObject*)out_obj;

    fftw_plan plan;

    const int ndim = PyArray_NDIM(in);
    int dims_in[ndim];
    int dims_out[ndim];

    void *indata = PyArray_DATA(in);
    void *outdata = PyArray_DATA(out);

    for(int i=0; i < ndim; i++) {
        dims_in[i] = (int)PyArray_DIMS(in)[i];
        dims_out[i] = (int)PyArray_DIMS(out)[i];
    }

    // NB: For real FFT the real array won't have flags.c_contiguous set, because of how
    // Numpy interprets the padding + slicing. Not sure if this is a good thing?
    // In any case we only assert contiguity of the complex array here.

    if (PyArray_DESCR(in)->type_num == NPY_DOUBLE)
    {
        assert_compatible_dtype(in, double);
        assert_compatible_dtype(out, double_complex);

        assert(PyArray_IS_C_CONTIGUOUS(out));

        plan = fftw_plan_dft_r2c(ndim, dims_in,
                                  (double *)indata,
                                  (fftw_complex *)outdata,
                                  flags);
    }
    else if (PyArray_DESCR(out)->type_num == NPY_DOUBLE)
    {
        assert_compatible_dtype(in, double_complex);
        assert_compatible_dtype(out, double);

        assert(PyArray_IS_C_CONTIGUOUS(in));

        plan = fftw_plan_dft_c2r(ndim, dims_out,
                                  (fftw_complex *)indata,
                                  (double *)outdata,
                                  flags);
    }
    else
    {
        assert_compatible_dtype(in, double_complex);
        assert_compatible_dtype(out, double_complex);

        assert(PyArray_IS_C_CONTIGUOUS(in));
        assert(PyArray_IS_C_CONTIGUOUS(out));

        plan = fftw_plan_dft(ndim, dims_out,
                              (fftw_complex *)indata,
                              (fftw_complex *)outdata,
                              sign, flags);
    }

    return PyLong_FromVoidPtr((void*)plan);
}


PyObject * FFTWExecute(PyObject *self, PyObject *args)
{
    PyObject* plan_obj;
    if (!PyArg_ParseTuple(args, "O", &plan_obj))
    {
        return NULL;
    }

    fftw_plan plan = (fftw_plan)PyLong_AsVoidPtr(plan_obj);
    fftw_execute(plan);
    Py_RETURN_NONE;
}


PyObject * FFTWDestroy(PyObject *self, PyObject *args)
{
    PyObject* plan_obj;
    if (!PyArg_ParseTuple(args, "O", &plan_obj))
    {
        return NULL;
    }

    fftw_plan plan = (fftw_plan)PyLong_AsVoidPtr(plan_obj);
    fftw_destroy_plan(plan);
    Py_RETURN_NONE;
}

#endif // GPAW_WITH_FFTW
