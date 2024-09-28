from __future__ import annotations
import contextlib
from time import time
from typing import TYPE_CHECKING
from types import ModuleType
from collections.abc import Iterable

import numpy as np

cupy_is_fake = True
"""True if :mod:`cupy` has been replaced by ``gpaw.gpu.cpupy``"""

is_hip = False
"""True if we are using HIP"""

device_id = None
"""Device id"""

if TYPE_CHECKING:
    import gpaw.gpu.cpupy as cupy
    import gpaw.gpu.cpupyx as cupyx
else:
    try:
        import gpaw.cgpaw as cgpaw
        if not hasattr(cgpaw, 'gpaw_gpu_init'):
            raise ImportError

        import cupy

        # This import is to preload cublas
        # Fixes cp.cublas.gemm attribute not found error introduced by v13.
        from cupy import cublas  # noqa: F401

        import cupyx
        from cupy.cuda import runtime
        numpy2 = np.__version__.split('.')[0] == '2'

        def fftshift_patch(x, axes=None):
            x = cupy.asarray(x)
            if axes is None:
                axes = list(range(x.ndim))
            elif not isinstance(axes, Iterable):
                axes = (axes,)
            return cupy.roll(x, [x.shape[axis] // 2 for axis in axes], axes)

        def ifftshift_patch(x, axes=None):
            x = cupy.asarray(x)
            if axes is None:
                axes = list(range(x.ndim))
            elif not isinstance(axes, Iterable):
                axes = (axes,)
            return cupy.roll(x, [-(x.shape[axis] // 2) for axis in axes], axes)

        if numpy2:
            cupy.fft.fftshift = fftshift_patch
            cupy.fft.ifftshift = ifftshift_patch

        is_hip = runtime.is_hip
        cupy_is_fake = False

        # Check the number of devices
        # Do not fail when calling `gpaw info` on a login node without GPUs
        try:
            device_count = runtime.getDeviceCount()
        except runtime.CUDARuntimeError as e:
            # Likely no device present
            if 'ErrorNoDevice' not in str(e):
                # Raise error in case of some other error
                raise e
            device_count = 0

        if device_count > 0:
            # select GPU device (round-robin based on MPI rank)
            # if not set, all MPI ranks will use the same default device
            from gpaw.mpi import rank
            runtime.setDevice(rank % device_count)

            # initialise C parameters and memory buffers
            import gpaw.cgpaw as cgpaw
            cgpaw.gpaw_gpu_init()

            # Generate a device id
            import os
            nodename = os.uname()[1]
            bus_id = runtime.deviceGetPCIBusId(runtime.getDevice())
            device_id = f'{nodename}:{bus_id}'

    except ImportError:
        import gpaw.gpu.cpupy as cupy
        import gpaw.gpu.cpupyx as cupyx

__all__ = ['cupy', 'cupyx', 'as_xp', 'as_np', 'synchronize']


def synchronize():
    if not cupy_is_fake:
        cupy.cuda.get_current_stream().synchronize()


def as_np(array: np.ndarray | cupy.ndarray) -> np.ndarray:
    """Transfer array to CPU (if not already there).

    Parameters
    ==========
    array:
        Numpy or CuPy array.
    """
    if isinstance(array, np.ndarray):
        return array
    return cupy.asnumpy(array)


def as_xp(array, xp):
    """Transfer array to CPU or GPU (if not already there).

    Parameters
    ==========
    array:
        Numpy or CuPy array.
    xp:
        :mod:`numpy` or :mod:`cupy`.
    """
    if xp is np:
        if isinstance(array, np.ndarray):
            return array
        return cupy.asnumpy(array)
    if isinstance(array, np.ndarray):
        return cupy.asarray(array)
    1 / 0
    return array


def einsum(subscripts, *operands, out):
    if isinstance(out, np.ndarray):
        np.einsum(subscripts, *operands, out=out)
    else:
        out[:] = cupy.einsum(subscripts, *operands)


def cupy_eigh(a: cupy.ndarray, UPLO: str) -> tuple[cupy.ndarray, cupy.ndarray]:
    """Wrapper for ``eigh()``.

    HIP-GPU version is too slow for now so we do it on the CPU.
    """
    from scipy.linalg import eigh
    if not is_hip:
        return cupy.linalg.eigh(a, UPLO=UPLO)
    eigs, evals = eigh(cupy.asnumpy(a),
                       lower=(UPLO == 'L'),
                       check_finite=False)

    return cupy.asarray(eigs), cupy.asarray(evals)


class XP:
    """Class for adding xp attribute (numpy or cupy).

    Also implements pickling which will not work out of the box
    because a module can't be pickled.
    """
    def __init__(self, xp: ModuleType):
        self.xp = xp

    def __getstate__(self):
        state = self.__dict__.copy()
        assert self.xp is np
        del state['xp']
        return state

    def __setstate__(self, state):
        state['xp'] = np
        self.__dict__.update(state)


@contextlib.contextmanager
def T():
    t1 = time()
    yield
    synchronize()
    t2 = time()
    print(f'{(t2 - t1) * 1e9:_.3f} ns')
