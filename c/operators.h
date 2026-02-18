#ifndef __OPERATORS_H
#define __OPERATORS_H

/*  Copyright (C) 2009-2012  CSC - IT Center for Science Ltd.
 *  Please see the accompanying LICENSE file for further information. */

#include "python_utils.h"
#include "bc.h"

#ifdef GPAW_GPU
#include "gpu/bmgs.h"
#endif

typedef struct
{
  PyObject_HEAD
  bmgsstencil stencil;
  boundary_conditions* bc;
  MPI_Request recvreq[2];
  MPI_Request sendreq[2];
  int nthreads;
#ifdef GPAW_GPU
  int use_gpu;
  bmgsstencil_gpu stencil_gpu;
#endif
} OperatorObject;

#ifdef GPAW_GPU
void operator_init_gpu(OperatorObject *self);
void operator_dealloc_gpu(int force);
#endif

void apply_worker(OperatorObject *self, int chunksize, int start,
		  int end, int thread_id, int nthreads,
		  const double* in, double* out,
		  bool real, const double_complex* ph);

#endif
