import numpy as np
import time as time

from gpaw.core.matrix import Matrix
from gpaw.core.plane_waves import PWArray, PWDesc
from gpaw.mpi import world, serial_comm


def test_array_me(N=20, max_mem=2e3, use_func=True):
    pw_desc = PWDesc(ecut=N**2 / 800,
                     cell=[10, 10, 10],  # bohr
                     kpt=[0, 0, 0],  # in units of reciprocal cell
                     comm=serial_comm,
                     dtype=np.complex128)
    psit_nX = PWArray(pw=pw_desc,
                      dims=(N,),
                      comm=world,
                      data=None,
                      xp=np)
    psit_nX.data[:] = 1
    M_nn = Matrix(
        N,
        N,
        dtype=np.complex128,
        data=None,
        dist=(world, -1, 1),
        xp=np,
    )
    M_nn.data[:] = 1

    if use_func:
        def func(psit_nX, out):
            # Do some random arithmetic
            out.data[:] = np.sqrt(psit_nX.data[:] * 2 + 100)**2
            out.data[:] = 1
            return out
        symmetric = True
    else:
        func = None
        symmetric = False

    buffer_mX = psit_nX.new_buffer(
        np.empty(int(max_mem), np.byte)
    )
    then = time.time()
    psit_nX.matrix_elements(psit_nX, function=func, out=M_nn,
                            sliced=True, symmetric=symmetric,
                            buffer=buffer_mX)
    now = time.time()
    assert np.allclose(M_nn.data, pw_desc.shape[0] * psit_nX.dv), \
        f'''Max: {np.max(M_nn.data)},
Min: {np.min(M_nn.data)},
Expected: {pw_desc.shape[0] * psit_nX.dv}'''
    return now - then


if __name__ == '__main__':
    max_mem_l = [2e7, 1e8, 2e8, 2e10]
    print([test_array_me(N=200, max_mem=max_mem, use_func=False)
           for max_mem in max_mem_l])
