#ifdef GPAW_WITH_FFTW
#include "python_utils.h"
#include <fftw3.h>

#include "gpaw_complex.h"

#define assert_compatible_dtype(arr, ctype) \
    assert(PyArray_ITEMSIZE(arr) == sizeof(ctype));


/* Create plan and return the plan handle as Python integer */
PyObject * FFTWPlan(PyObject *self, PyObject *args)
{
    PyArrayObject* in_obj;
    PyArrayObject* out_obj;
    int sign;
    unsigned int flags;
    if (!PyArg_ParseTuple(args, "OOiI", &in_obj, &out_obj, &sign, &flags))
    {
        return NULL;
    }

    assert(PyArray_Check(in_obj));
    assert(PyArray_Check(out_obj));

    fftw_plan plan;

    const int ndim = PyArray_NDIM(in_obj);
    int dims_in[ndim];
    int dims_out[ndim];

    void *indata = PyArray_DATA(in_obj);
    void *outdata = PyArray_DATA(out_obj);

    for(int i=0; i < ndim; i++) {
        dims_in[i] = (int)PyArray_DIMS(in_obj)[i];
        dims_out[i] = (int)PyArray_DIMS(out_obj)[i];
    }

    // NB: For real FFT the real array won't have flags.c_contiguous set, because of how
    // Numpy interprets the padding + slicing. Not sure if this is a good thing?
    // In any case we only assert contiguity of the complex array here.

    if (PyArray_DESCR(in_obj)->type_num == NPY_DOUBLE)
    {
        assert_compatible_dtype(in_obj, double);
        assert_compatible_dtype(out_obj, double_complex);

        assert(PyArray_IS_C_CONTIGUOUS(out_obj));

        plan = fftw_plan_dft_r2c(ndim, dims_in,
                                  (double *)indata,
                                  (fftw_complex *)outdata,
                                  flags);
    }
    else if (PyArray_DESCR(out_obj)->type_num == NPY_DOUBLE)
    {
        assert_compatible_dtype(in_obj, double_complex);
        assert_compatible_dtype(out_obj, double);

        assert(PyArray_IS_C_CONTIGUOUS(in_obj));

        plan = fftw_plan_dft_c2r(ndim, dims_out,
                                  (fftw_complex *)indata,
                                  (double *)outdata,
                                  flags);
    }
    else
    {
        assert_compatible_dtype(in_obj, double_complex);
        assert_compatible_dtype(out_obj, double_complex);

        assert(PyArray_IS_C_CONTIGUOUS(in_obj));
        assert(PyArray_IS_C_CONTIGUOUS(out_obj));

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
