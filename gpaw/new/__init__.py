"""New ground-state DFT code."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from contextlib import contextmanager
from time import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpaw.core import UGArray

from gpaw.new.timer import trace, tracectx  # noqa


def prod(iterable: Iterable[int]) -> int:
    """Simple int product.

    >>> prod([])
    1
    >>> prod([2, 3])
    6
    """
    result = 1
    for x in iterable:
        result *= x
    return result


def zips(*iterables, strict=True):
    yield from zip(*iterables, strict=strict)


def spinsum(a_sX: UGArray, mean: bool = False) -> UGArray:
    if a_sX.dims[0] == 2:
        a_X = a_sX.desc.empty(xp=a_sX.xp)
        a_sX.data[:2].sum(axis=0, out=a_X.data)
        if mean:
            a_X.data *= 0.5
        return a_X
    return a_sX[0]


class Timer:
    def __init__(self):
        self.times = defaultdict(float)
        self.times['Total'] = -time()

    @contextmanager
    def __call__(self, name):
        t1 = time()
        try:
            yield
        finally:
            t2 = time()
            self.times[name] += t2 - t1

    def write(self, log):
        self.times['Total'] += time()
        total = self.times['Total']
        log('\ntiming:  # [seconds]')
        n = max(len(name) for name in self.times) + 2
        w = len(f'{total:.3f}')
        N = 71 - n - w
        for name, t in self.times.items():
            m = int(round(2 * N * t / total))
            bar = '━' * (m // 2) + '╸' * (m % 2)
            log(f'  {name + ":":{n}}{t:{w}.3f}  # {bar}')
