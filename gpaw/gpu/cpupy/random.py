import numpy as np


def default_rng(seed):
    return RNG(np.random.default_rng(seed))


class RNG:
    def __init__(self, rng):
        self.rng = rng

    def random(self, shape=None, dtype=float, out=None):
        from gpaw.gpu.cpupy import ndarray
        return ndarray(self.rng.random(shape, dtype=dtype, out=out))
