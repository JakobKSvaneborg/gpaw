from __future__ import annotations

import warnings
from functools import partial
from typing import Callable

import numpy as np
from ase.units import Ha

from gpaw.core.arrays import DistributedArrays as XArray
from gpaw.core.atom_centered_functions import AtomArrays
from gpaw.mpi import broadcast_exception, broadcast_float
from gpaw.new import trace, zips
from gpaw.new.c import calculate_residuals_gpu
from gpaw.new.eigensolver import Eigensolver, calculate_weights
from gpaw.new.energies import DFTEnergies
from gpaw.new.hamiltonian import Hamiltonian
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from gpaw.typing import Array1D
from gpaw.utilities.blas import axpy
from gpaw.utilities import as_real_dtype


def create_eigensolver(nbands,
                       wf_desc,
                       band_comm,
                       comm,
                       create_preconditioner,
                       converge_bands,
                       setups,
                       atoms,
                       name='dav',
                       **kwargs):
    if name in ['rmm-diis', 'cg', 'direct']:
        warnings.warn(f'{name} not implemented.  Using dav instead')
        name = 'dav'
    if name == 'dav':
        from gpaw.new.pwfd.davidson import Davidson
        return Davidson(
            nbands,
            wf_desc,
            band_comm,
            create_preconditioner,
            converge_bands,
            **kwargs)
    if name == 'rmm-diis':
        from gpaw.new.pwfd.rmmdiis import RMMDIIS
        return RMMDIIS(
            nbands,
            wf_desc,
            band_comm,
            create_preconditioner,
            converge_bands,
            **kwargs)
    if name == 'etdm-fdpw':
        from gpaw.new.pwfd.etdm import ETDM
        return ETDM(**kwargs)
    raise ValueError


class PWFDEigensolver(Eigensolver):
    def __init__(self,
                 preconditioner_factory,
                 converge_bands='occupied',
                 blocksize=10):
        self.converge_bands = converge_bands
        self.blocksize = blocksize
        self.preconditioner = None
        self.preconditioner_factory = preconditioner_factory

    def _initialize(self, ibzwfs):
        # First time: allocate work-arrays
        wfs = ibzwfs.wfs_qs[0][0]
        assert isinstance(wfs, PWFDWaveFunctions)
        xp = wfs.psit_nX.xp
        self.preconditioner = self.preconditioner_factory(self.blocksize,
                                                          xp=xp)

    @trace
    def iterate(self,
                ibzwfs,
                density,
                potential,
                hamiltonian: Hamiltonian,
                pot_calc,
                energies: DFTEnergies) -> tuple[float, DFTEnergies]:
        """Iterate on state given fixed hamiltonian.

        Returns
        -------
        float:
            Weighted error of residuals:::

                   ~     ~ ~
              R = (H - ε S)ψ
               n        n   n
        """

        if self.preconditioner is None:
            self._initialize(ibzwfs)

        wfs = ibzwfs.wfs_qs[0][0]
        dS_aii = wfs.setups.get_overlap_corrections(wfs.P_ani.layout.atomdist,
                                                    wfs.xp)
        dH = potential.dH
        Ht = partial(hamiltonian.apply,
                     potential.vt_sR,
                     potential.dedtaut_sR,
                     ibzwfs, density.D_asii)  # used by hybrids

        weight_un = calculate_weights(self.converge_bands, ibzwfs)

        error = 0.0
        eig_error = 0.0
        # Loop over k-points:
        with broadcast_exception(ibzwfs.kpt_comm):
            for wfs, weight_n in zips(ibzwfs, weight_un):
                eig_old = wfs.myeig_n
                e = self.iterate1(wfs, Ht, dH, dS_aii, weight_n)
                error += wfs.weight * e
                occs = wfs.myocc_n
                eig_error += weight_n @ np.abs(eig_old - wfs.myeig_n)**2

        error = ibzwfs.kpt_band_comm.sum_scalar(
            float(error)) * ibzwfs.spin_degeneracy
        eig_error = (ibzwfs.kpt_band_comm.sum_scalar(
                     float(eig_error)) * ibzwfs.spin_degeneracy)**0.5

        return eig_error, error, energies

    def iterate1(self, wfs, Ht, dH, dS_aii, weight_n):
        raise NotImplementedError


@trace
def calculate_residuals(residual_nX: XArray,
                        dH: Callable[[AtomArrays, AtomArrays], AtomArrays],
                        dS_aii: AtomArrays,
                        wfs: PWFDWaveFunctions,
                        P1_ani: AtomArrays,
                        P2_ani: AtomArrays) -> None:

    eig_n = wfs.myeig_n
    xp = residual_nX.xp
    if xp is np:
        for r, e, p in zips(residual_nX.data, eig_n, wfs.psit_nX.data):
            axpy(-e, p, r)
    else:
        eig_n = xp.asarray(eig_n, dtype=as_real_dtype(residual_nX.data.dtype))
        calculate_residuals_gpu(residual_nX.data, eig_n, wfs.psit_nX.data)

    dH(wfs.P_ani, P1_ani)
    wfs.P_ani.block_diag_multiply(dS_aii, out_ani=P2_ani)

    if wfs.ncomponents < 4:
        subscripts = 'nI, n -> nI'
    else:
        subscripts = 'nsI, n -> nsI'
    if xp is np:
        np.einsum(subscripts, P2_ani.data, eig_n, out=P2_ani.data,
                  dtype=P2_ani.data.dtype, casting='same_kind')
    else:
        P2_ani.data[:] = xp.einsum(subscripts, P2_ani.data, eig_n)
    P1_ani.data -= P2_ani.data
    wfs.pt_aiX.add_to(residual_nX, P1_ani)

