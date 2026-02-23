# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.
from time import perf_counter as clock

from gpaw.old.grid_descriptor import GridDescriptor
from gpaw.transformers import Transformer


def test_timing(mpi):
    n = 6
    gda = GridDescriptor((n, n, n), comm=mpi.comm)
    gdb = gda.refine()
    gdc = gdb.refine()
    a = gda.zeros()
    b = gdb.zeros()
    c = gdc.zeros()

    inter = Transformer(gdb, gdc, 2).apply
    restr = Transformer(gdb, gda, 2).apply

    t = clock()
    for i in range(8 * 300):
        inter(b, c)
    print(clock() - t)

    t = clock()
    for i in range(8 * 3000):
        restr(b, a)
    print(clock() - t)
