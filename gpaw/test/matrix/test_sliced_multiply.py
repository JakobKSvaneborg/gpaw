import pytest
import numpy as np
import time
from gpaw.utilities.blas import mmm
from gpaw.mpi import world

def test_sliced_multiply(N=10, max_mem=2e2):
    from gpaw.core.matrix import Matrix
    
    A_nn = Matrix(
        N,
        N,
        dtype=np.complex128,
        data=None,
        dist=(world, world.size, 1),
        xp=np,
    )
    B_nX = Matrix(
        N,
        100*N,
        dtype=np.complex128,
        data=None,
        dist=(world, world.size, 1),
        xp=np,
    )
    
    dist = B_nX.dist
    dtype = B_nX.dtype
    
    # allocate buffers for Ht @ psit
    buffer_size = max(
        min(int(dist.comm.size * max_mem /
                (max(N, 1) *
                        dtype.itemsize)),
            B_nX.data.shape[1]), 1)
    buffer_nx = Matrix(M=N, N=buffer_size,
                       dtype=dtype,
                       dist=(dist.new(M=B_nX.shape[0],
                                      N=buffer_size)),
                       xp=B_nX.xp)
    print(buffer_nx.data.shape)
    buffers_nx = [buffer_nx, buffer_nx.new()]
    if buffer_nx.shape == B_nX.shape:
        # Only time the multiply
        then = time.time()
        #mmm(1.0, A_nn.data, 'N', B_nX.data, 'N', 0.0, buffer_nx.data)
        #mmm(1.0, A_nn.data, 'N', B_nX.data, 'N', 0.0, buffer_nx.data)
        #mmm(1.0, A_nn.data, 'N', B_nX.data, 'N', 0.0, buffer_nx.data)
        
        A_nn.multiply(B_nX, out=buffer_nx)
        A_nn.multiply(B_nX, out=buffer_nx)
        A_nn.multiply(B_nX, out=buffer_nx)
        #buffer_nx.data[:] = A_nn.data @ B_nX.data
        #buffer_nx.data[:] = A_nn.data @ B_nX.data
        #buffer_nx.data[:] = A_nn.data @ B_nX.data
        
        now = time.time()
        return(now - then)

    else:
        # Only time the multiply
        then = time.time()
        A_nn.multiply(B_nX, out=B_nX, buffers=buffers_nx)
        A_nn.multiply(B_nX, out=B_nX, buffers=buffers_nx)
        A_nn.multiply(B_nX, out=B_nX, buffers=buffers_nx)
        now = time.time()
        return(now - then)

if __name__ == '__main__':
    max_mem_l = [2e5, 2e6, 2e7, 2e10]
    times = [test_sliced_multiply(N=500, max_mem=mem) for mem in max_mem_l]
    print(max_mem_l, times)
