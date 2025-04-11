import numpy as np
import time as time

from functools import partial

from gpaw.core.matrix import Matrix
from gpaw.core.plane_waves import PWArray, PWDesc
from gpaw.mpi import world, serial_comm


def test_array_me(N=50, max_mem=2e2):
    pw_desc = PWDesc(ecut=N/4,
                     cell=[10, 10, 10],  # bohr
                     kpt=[0, 0, 0],  # in units of reciprocal cell
                     comm=serial_comm,
                     dtype=np.complex128)
    psit_nX = PWArray(pw=pw_desc,
                      dims=(N,),
                      comm=world,
                      data=None,
                      xp=np)
    M_nn = Matrix(
        N,
        N,
        dtype=np.complex128,
        data=None,
        dist=(world, world.size, 1),
        xp=np,
    )
    
    def func(psit_nX, out):
        # Do some random arithmetic
        out.data[:] = np.sqrt(psit_nX.data[:] * 2 + 100)**2
        return out
    
    buffer_mx = psit_nX.get_buffer(max_mem=max_mem)[0]
    print(buffer_mx.data.shape)
    if buffer_mx.matrix.shape == psit_nX.matrix.shape:
        then = time.time()
        psit_nX.matrix_elements(psit_nX, function=partial(func, out=buffer_mx), out=M_nn, sliced=False, symmetric=True)
        now = time.time()
    else:
        then = time.time()
        psit_nX.matrix_elements(psit_nX, function=func, out=M_nn, sliced=True)
        now = time.time()
    return now - then

if __name__ == '__main__':
    max_mem_l = [2e5, 2e6, 2e7, 2e8, 2e10]
    print([test_array_me(N=500, max_mem=max_mem) for max_mem in max_mem_l])