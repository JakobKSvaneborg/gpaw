
#include "python_utils.h"
#include "gpaw_utils.h"

#ifdef PARALLEL
#include <mpi.h>
#endif

#include <stdio.h>


void gpaw_set_runtime_error(const char* err_msg)
{
    PyErr_SetString(PyExc_RuntimeError, err_msg);
}

CLINKAGE void gpaw_abort(const char* err_msg)
{
    printf("\nFatal GPAW error: %s\n", err_msg);
    fflush(stdout);

    gpaw_set_runtime_error(err_msg);

#ifdef PARALLEL
    MPI_Abort(MPI_COMM_WORLD, -1);
#endif
}
