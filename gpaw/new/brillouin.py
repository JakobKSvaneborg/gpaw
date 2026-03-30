"""Brillouin-zone sampling."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ase.dft.kpoints import monkhorst_pack

from gpaw.mpi import MPIComm
from gpaw.symmetry import reduce_kpts
from gpaw.typing import Array1D, Array2D, ArrayLike2D

if TYPE_CHECKING:
    from gpaw.new.symmetry import Symmetries


class BZPoints:
    def __init__(self, points: ArrayLike2D):
        self.kpt_Kc = np.array(points)
        assert self.kpt_Kc.ndim == 2
        assert self.kpt_Kc.shape[1] == 3
        self.gamma_only = len(self.kpt_Kc) == 1 and not self.kpt_Kc.any()

    def __len__(self):
        """Number of k-points in the BZ."""
        return len(self.kpt_Kc)

    def __repr__(self):
        if self.gamma_only:
            return 'BZPoints([<gamma only>])'
        return f'BZPoints([<{len(self)} points>])'

    def reduce(self,
               symmetries: Symmetries,
               *,
               comm: MPIComm = None,
               strict: bool = True,
               use_time_reversal=True,
               tolerance=1e-7) -> IBZ:
        """Find irreducible set of k-points."""
        if not use_time_reversal and len(symmetries) == 1:
            N = len(self)
            return IBZ(symmetries,
                       self,
                       ibz2bz=np.arange(N),
                       bz2ibz=np.arange(N),
                       weights=np.ones(N) / N,
                       bz2bz_Ks=np.arange(N).reshape((N, 1)),
                       s_K=np.zeros(N, int),
                       time_reversal_K=np.zeros(N, bool))

        if symmetries.has_inversion:
            use_time_reversal = False
        (_, weight_k, sym_K, time_reversal_K, bz2ibz_K, ibz2bz_k,
         bz2bz_Ks) = reduce_kpts(self.kpt_Kc,
                                 symmetries.rotation_scc,
                                 use_time_reversal,
                                 comm,
                                 tolerance)
        assert (weight_k > 0.0).all()

        if strict and -1 in bz2bz_Ks:
            raise ValueError(
                'Your k-points are not as symmetric as your crystal!')

        return IBZ(symmetries, self, ibz2bz_k, bz2ibz_K, weight_k, bz2bz_Ks,
                   sym_K, time_reversal_K)


class BZBandPath(BZPoints):
    def __init__(self, band_path):
        self.band_path = band_path
        super().__init__(band_path.kpts)


class MonkhorstPackKPoints(BZPoints):
    def __init__(self, size, shift=(0, 0, 0)):
        self.size_c = size
        self.shift_c = np.array(shift)
        super().__init__(monkhorst_pack(size) + shift)

    def __repr__(self):
        return f'MonkhorstPackKPoints({self.size_c}, shift={self.shift_c})'

    def __str__(self):
        a, b, c = self.size_c
        l, m, n = self.shift_c
        return (f'Monkhorst-Pack size: [{a}, {b}, {c}]\n'
                f'Monkhorst-Pack shift: [{l}, {m}, {n}]\n')


class IBZ:
    def __init__(self,
                 symmetries: Symmetries,
                 bz: BZPoints,
                 ibz2bz, bz2ibz, weights,
                 bz2bz_Ks=None, s_K=None, time_reversal_K=None):
        self.symmetries = symmetries
        self.bz = bz
        self.weight_k = weights
        self.kpt_kc = bz.kpt_Kc[ibz2bz]
        self.ibz2bz_k = ibz2bz
        self.bz2ibz_K = bz2ibz
        self.bz2bz_Ks = bz2bz_Ks
        self.s_K = s_K
        self.time_reversal_K = time_reversal_K

    def __len__(self):
        """Number of k-points in the IBZ."""
        return len(self.kpt_kc)

    def __repr__(self):
        return (f'IBZ(<points: {len(self)}, '
                f'symmetries: {len(self.symmetries)}>)')

    def __str__(self):
        N = len(self)
        txt = ('BZ-sampling:\n'
               f'  Number of bz points: {len(self.bz)}\n'
               f'  Number of ibz points: {N}\n')

        if self.bz2bz_Ks is not None and -1 in self.bz2bz_Ks:
            txt += '  Your k-points are not as symmetric as your crystal!\n'

        if isinstance(self.bz, MonkhorstPackKPoints):
            txt += '  ' + str(self.bz).replace('\n', '\n  ', 1)

        txt += (
            '  Points  # in reciprocal-cell coordinates\n'
            '  ----------------------------------------------------------\n'
            '                      coordinates                     weight\n'
            '  ----------------------------------------------------------\n')
        k = 0
        while k < N:
            if k == 10:
                if N > 10:
                    txt += '  ...\n'
                k = N - 1
            a, b, c = self.kpt_kc[k]
            w = self.weight_k[k]
            txt += (f'  {k:4} ({a:12.8f}, {b:12.8f}, {c:12.8f}) '
                    f'{w:.8f}\n')
            k += 1
        txt += (
            '  ----------------------------------------------------------\n')
        return txt

    def ranks(self, comm: MPIComm, nspins: int = 1) -> Array2D:
        """Distribute k-points over MPI-communicator."""
        return ranks(comm.size, len(self) * nspins).reshape((-1, nspins))

    def _old_kd(self, nspins, kpt_comm):
        from gpaw.old.kpt_descriptor import KPointDescriptor
        kd = KPointDescriptor(self.bz.kpt_Kc, nspins)
        kd.ibzk_kc = self.kpt_kc
        kd.weight_k = self.weight_k
        kd.sym_k = self.s_K
        kd.time_reversal_k = self.time_reversal_K
        kd.bz2ibz_k = self.bz2ibz_K
        kd.ibz2bz_k = self.ibz2bz_k
        kd.bz2bz_ks = self.bz2bz_Ks
        kd.nibzkpts = len(self)
        kd.symmetry = self.symmetries._old_symmetry
        kd.set_communicator(kpt_comm)
        rank_ks = self.ranks(kpt_comm, nspins)
        here_k = (rank_ks == kpt_comm.rank).any(axis=1)
        kd.ibzk_qc = self.kpt_kc[here_k]
        kd.rank0 = 'hello'
        kd.mynk = 'hello'
        kd.k0 = 'hello'
        kd.weight_q = -5555555555
        kd.nu_r = np.zeros(kpt_comm.size, int)
        kd.nu_r[kpt_comm.rank] = (rank_ks == kpt_comm.rank).sum()
        kpt_comm.sum(kd.nu_r)
        kd.rank_ks = rank_ks
        return kd


def ranks(N, K) -> Array1D:
    """Distribute k-points over MPI-communicator.

    >>> ranks(4, 6)
    array([0, 1, 2, 2, 3, 3])
    """
    n, x = divmod(K, N)
    rnks = np.empty(K, int)
    r = N - x
    for k in range(r * n):
        rnks[k] = k // n
    for k in range(r * n, K):
        rnks[k] = (k - r * n) // (n + 1) + r
    return rnks
