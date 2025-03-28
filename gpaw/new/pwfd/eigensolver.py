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
from gpaw.new.eigensolver import Eigensolver
from gpaw.new.energies import DFTEnergies
from gpaw.new.hamiltonian import Hamiltonian
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.typing import Array1D
from gpaw.utilities.blas import axpy


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
    if name in ['cg', 'direct']:
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
        self.preconditioner = self.preconditioner_factory(self.blocksize,
                                                          xp=ibzwfs.xp)

    def _allocate_work_arrays(self, ibzwfs, shape):
        b = max(wfs.n2 - wfs.n1 for wfs in ibzwfs)
        shape += (b,) + ibzwfs.get_max_shape()
        dtype = ibzwfs.wfs_qs[0][0].psit_nX.data.dtype
        self.work_arrays = ibzwfs.xp.empty(shape, dtype)

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

                           ~
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
        # Loop over k-points:
        with broadcast_exception(ibzwfs.kpt_comm):
            for wfs, weight_n in zips(ibzwfs, weight_un):
                e = self.iterate1(wfs, Ht, dH, dS_aii, weight_n)
                error += wfs.weight * e

        error = ibzwfs.kpt_band_comm.sum_scalar(
            float(error)) * ibzwfs.spin_degeneracy

        return error, energies

    def iterate1(self, wfs, Ht, dH, dS_aii, weight_n):
        raise NotImplementedError


@trace
def calculate_residuals(psit_nX,
                        residual_nX: XArray,
                        pt_aiX,
                        P_ani,
                        eig_n,
                        dH: Callable[[AtomArrays, AtomArrays], AtomArrays],
                        dS_aii: AtomArrays,
                        P1_ani: AtomArrays,
                        P2_ani: AtomArrays) -> None:
    """Complete the calculation of resuduals.

    Starting from residual_nX having the values:::

       ^   ~  ~
      (T + v) ψ
               n

    add the following:::

      --   a       a   ~a ~   ~a    ~
      > (ΔH  - ε ΔS  )<p |ψ > p - ε ψ .
      --   ij   n  ij   j  n   i     n
      ij

    (P1_ani and P2_ani are work buffers).
    """
    xp = residual_nX.xp
    if xp is np:
        for r, e, p in zips(residual_nX.data, eig_n, psit_nX.data):
            axpy(-e, p, r)
    else:
        eig_n = xp.asarray(eig_n)
        calculate_residuals_gpu(residual_nX.data, eig_n, psit_nX.data)

    dH(P_ani, P1_ani)
    P_ani.block_diag_multiply(dS_aii, out_ani=P2_ani)

    if P_ani.data.ndim == 2:
        subscripts = 'nI, n -> nI'
    else:
        subscripts = 'nsI, n -> nsI'
    if xp is np:
        np.einsum(subscripts, P2_ani.data, eig_n, out=P2_ani.data,
                  dtype=P2_ani.data.dtype, casting='same_kind')
    else:
        P2_ani.data[:] = xp.einsum(subscripts, P2_ani.data, eig_n)
    P1_ani.data -= P2_ani.data
    pt_aiX.add_to(residual_nX, P1_ani)


def calculate_weights(converge_bands: int | str,
                      ibzwfs: IBZWaveFunctions) -> list[Array1D | None]:
    """Calculate convergence weights for all eigenstates."""
    weight_un = []
    nu = len(ibzwfs.wfs_qs) * ibzwfs.nspins
    nbands = ibzwfs.nbands

    if converge_bands == 'occupied':
        # Converge occupied bands:
        for wfs in ibzwfs:
            try:
                # Methfessel-Paxton or cold-smearing distributions can give
                # negative occupation numbers - so we take the absolute value:
                weight_n = np.abs(wfs.myocc_n)
            except ValueError:
                # No eigenvalues yet:
                return [None] * nu
            weight_un.append(weight_n)
        return weight_un

    if converge_bands == 'all':
        converge_bands = nbands

    if not isinstance(converge_bands, str):
        # Converge fixed number of bands:
        n = converge_bands
        if n < 0:
            n += nbands
            assert n >= 0
        for wfs in ibzwfs:
            weight_n = np.zeros(wfs.n2 - wfs.n1)
            m = max(wfs.n1, min(n, wfs.n2)) - wfs.n1
            weight_n[:m] = 1.0
            weight_un.append(weight_n)
        return weight_un

    # Converge states with energy up to CBM + delta:
    assert converge_bands.startswith('CBM+')
    delta = float(converge_bands[4:]) / Ha

    if ibzwfs.fermi_levels is None:
        return [None] * nu

    efermi = np.mean(ibzwfs.fermi_levels)

    # Find CBM:
    cbm = np.inf
    nocc_u = np.empty(nu, int)
    for u, wfs in enumerate(ibzwfs):
        n = (wfs.eig_n < efermi).sum()  # number of occupied bands
        nocc_u[u] = n
        if n < nbands:
            cbm = min(cbm, wfs.eig_n[n])

    # If all k-points don't have the same number of occupied bands,
    # then it's a metal:
    n0 = int(broadcast_float(float(nocc_u[0]), ibzwfs.kpt_comm))
    metal = bool(ibzwfs.kpt_comm.sum_scalar(float((nocc_u != n0).any())))
    if metal:
        cbm = efermi
    else:
        cbm = ibzwfs.kpt_comm.min_scalar(cbm)

    ecut = cbm + delta

    for wfs in ibzwfs:
        weight_n = (wfs.myeig_n < ecut).astype(float)
        if wfs.eig_n[-1] < ecut:
            # We don't have enough bands!
            weight_n[:] = np.inf
        weight_un.append(weight_n)

    return weight_un
