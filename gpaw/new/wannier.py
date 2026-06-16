from __future__ import annotations

from math import factorial as fac

import numpy as np
from ase.units import Bohr

from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.spline import Spline
from gpaw.typing import Array2D


def get_wannier_integrals(ibzwfs: IBZWaveFunctions,
                          grid,
                          s: int,
                          k: int,
                          k1: int,
                          G_c,
                          nbands=None) -> Array2D:
    """Calculate integrals for maximally localized Wannier functions.

    Supports k-point parallelization: when k and k1 reside on different
    MPI ranks, wavefunctions are communicated via ``kpt_comm`` before
    computing the overlap matrix.
    """
    ibzwfs.make_sure_wfs_are_read_from_gpw_file()
    assert s <= ibzwfs.nspins
    assert ibzwfs.band_comm.size == 1, \
        'Band parallelization is not supported for Wannier integrals'

    kpt_comm = ibzwfs.kpt_comm
    nb = nbands if nbands else ibzwfs.nbands

    if kpt_comm.size == 1:
        # No k-point parallelization - original fast path
        wfs = ibzwfs._get_wfs(k, s).to_uniform_grid_wave_functions(
            grid, None)
        wfs1 = ibzwfs._get_wfs(k1, s).to_uniform_grid_wave_functions(
            grid, None)
        Z_nn = grid._gd.wannier_matrix(
            wfs.psit_nX.data, wfs1.psit_nX.data, G_c, nb)
        add_wannier_correction(Z_nn, G_c, wfs, wfs1, nb)
        grid.comm.sum(Z_nn)
        return Z_nn

    # --- k-point parallel path ---
    rank_k = ibzwfs.rank_ks[k, s]
    rank_k1 = ibzwfs.rank_ks[k1, s]
    my_rank = kpt_comm.rank

    Z_nn = np.zeros((nb, nb), complex)

    if rank_k == rank_k1:
        # Both k-points on same kpt_comm rank
        if my_rank == rank_k:
            wfs = ibzwfs._get_wfs(k, s).to_uniform_grid_wave_functions(
                grid, None)
            wfs1 = ibzwfs._get_wfs(k1, s).to_uniform_grid_wave_functions(
                grid, None)
            Z_nn = grid._gd.wannier_matrix(
                wfs.psit_nX.data, wfs1.psit_nX.data, G_c, nb)
            add_wannier_correction(Z_nn, G_c, wfs, wfs1, nb)
            grid.comm.sum(Z_nn)
    else:
        # k-points on different kpt_comm ranks - communicate wfs for k1
        if my_rank == rank_k:
            wfs = ibzwfs._get_wfs(k, s).to_uniform_grid_wave_functions(
                grid, None)
            wfs1 = _receive_wannier_wfs(
                kpt_comm, rank_k1, grid, ibzwfs, k1, s)
            Z_nn = grid._gd.wannier_matrix(
                wfs.psit_nX.data, wfs1.psit_nX.data, G_c, nb)
            add_wannier_correction(Z_nn, G_c, wfs, wfs1, nb)
            grid.comm.sum(Z_nn)
        elif my_rank == rank_k1:
            _send_wannier_wfs(kpt_comm, rank_k, grid, ibzwfs, k1, s)

    kpt_comm.broadcast(Z_nn, rank_k)
    return Z_nn


def _send_wannier_wfs(kpt_comm, dest_rank, grid, ibzwfs, k1, s):
    """Send domain-decomposed wfs for k1 to dest_rank via kpt_comm."""
    wfs1 = ibzwfs._get_wfs(k1, s).to_uniform_grid_wave_functions(grid, None)
    kpt_comm.send(np.ascontiguousarray(wfs1.psit_nX.data), dest_rank)


def _receive_wannier_wfs(kpt_comm, src_rank, grid, ibzwfs, k1, s):
    """Receive domain-decomposed wfs for k1 from src_rank via kpt_comm.

    Reconstructs a PWFDWaveFunctions on the receiving side so that
    P_ani (PAW projections) is correctly computed lazily using k1's
    k-point-dependent projectors.
    """
    from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions

    # Create UG descriptor for k1 (preserves domain decomposition)
    kpt_c = ibzwfs.ibz.kpt_kc[k1]
    grid_k1 = grid.new(kpt=kpt_c, dtype=ibzwfs.dtype)

    # Receive raw wavefunction array (same local grid shape since
    # kpt_comm connects corresponding domain ranks)
    shape = (ibzwfs.nbands,) + tuple(grid_k1.mysize_c)
    psit1_data = np.empty(shape, dtype=ibzwfs.dtype)
    kpt_comm.receive(psit1_data, src_rank)

    # Wrap in UGArray via from_data (infers dims from shape)
    psit1_nR = grid_k1.from_data(psit1_data)

    # Construct PWFDWaveFunctions using shared metadata
    local_wfs = ibzwfs._wfs_u[0]
    wfs1 = PWFDWaveFunctions(
        psit1_nR,
        spin=s, q=0, k=k1,
        setups=local_wfs.setups,
        relpos_ac=local_wfs.relpos_ac,
        atomdist=local_wfs.atomdist,
        weight=1.0,
        ncomponents=local_wfs.ncomponents)
    return wfs1


def add_wannier_correction(Z_nn, G_c, wfs, wfs1, nbands):
    r"""Calculate the correction to the wannier integrals.

    See: (Eq. 27 ref1)::

                      -i G.r
        Z   = <psi | e      |psi >
         nm       n             m

                       __                __
               ~      \              a  \     a*   a    a
        Z    = Z    +  ) exp[-i G . R ]  )   P   dO    P
         nmx    nmx   /__            x  /__   ni   ii'  mi'

                       a                 ii'

    Note that this correction is an approximation that assumes the
    exponential varies slowly over the extent of the augmentation sphere.

    ref1: Thygesen et al, Phys. Rev. B 72, 125119 (2005)
    """
    P_ani = wfs.P_ani
    P1_ani = wfs1.P_ani
    for a, P_ni in P_ani.items():
        P_ni = P_ani[a][:nbands]
        P1_ni = P1_ani[a][:nbands]
        dO_ii = wfs.setups[a].dO_ii
        e = np.exp(-2.j * np.pi * np.dot(G_c, wfs.relpos_ac[a]))
        Z_nn += e * P_ni.conj() @ dO_ii @ P1_ni.T


def initial_wannier(ibzwfs: IBZWaveFunctions,
                    initialwannier, kpointgrid, fixedstates,
                    edf, spin, nbands):
    """Initial guess for the shape of wannier functions.

    Use initial guess for wannier orbitals to determine rotation
    matrices U and C.
    """

    from ase.dft.wannier import rotation_from_projection
    proj_knw = get_projections(ibzwfs, initialwannier, spin)
    U_kww = []
    C_kul = []
    for fixed, proj_nw in zip(fixedstates, proj_knw):
        U_ww, C_ul = rotation_from_projection(proj_nw[:nbands],
                                              fixed,
                                              ortho=True)
        U_kww.append(U_ww)
        C_kul.append(C_ul)

    return C_kul, np.asarray(U_kww)


def get_projections(ibzwfs: IBZWaveFunctions,
                    locfun: str | list[tuple],
                    spin=0):
    """Project wave functions onto localized functions

    Determine the projections of the Kohn-Sham eigenstates
    onto specified localized functions of the format::

      locfun = [[spos_c, l, sigma], [...]]

    spos_c can be an atom index, or a scaled position vector. l is
    the angular momentum, and sigma is the (half-) width of the
    radial gaussian.

    Return format is::

      f_kni = <psi_kn | f_i>

    where psi_kn are the wave functions, and f_i are the specified
    localized functions.

    As a special case, locfun can be the string 'projectors', in which
    case the bound state projectors are used as localized functions.
    """
    if isinstance(locfun, str):
        assert locfun == 'projectors'
        # Count number of bound-state projector functions
        nproj = 0
        for setup in ibzwfs._wfs_u[0].setups:
            for l, n in zip(setup.l_j, setup.n_j):
                if n >= 0:
                    nproj += 2 * l + 1

        nkpts = len(ibzwfs.ibz)
        f_kni = np.zeros((nkpts, ibzwfs.nbands, nproj), ibzwfs.dtype)
        for wfs in ibzwfs:
            if wfs.spin == spin:
                f_in = []
                for a, P_ni in wfs.P_ani.items():
                    i = 0
                    setup = wfs.setups[a]
                    for l, n in zip(setup.l_j, setup.n_j):
                        if n >= 0:
                            for j in range(i, i + 2 * l + 1):
                                f_in.append(P_ni[:, j])
                        i += 2 * l + 1
                f_kni[wfs.k] = np.array(f_in).T
        ibzwfs.kpt_comm.sum(f_kni)
        return f_kni.conj()

    nkpts = len(ibzwfs.ibz)
    nbf = np.sum([2 * l + 1 for pos, l, a in locfun])
    f_knB = np.zeros((nkpts, ibzwfs.nbands, nbf), ibzwfs.dtype)
    relpos_ac = ibzwfs._wfs_u[0].relpos_ac

    spos_bc = []
    splines_b = []
    for spos_c, l, sigma in locfun:
        if isinstance(spos_c, int):
            spos_c = relpos_ac[spos_c]
        spos_bc.append(spos_c)
        alpha = .5 * Bohr**2 / sigma**2
        r = np.linspace(0, 10. * sigma, 500)
        f_g = (fac(l) * (4 * alpha)**(l + 3 / 2.) *
               np.exp(-alpha * r**2) /
               (np.sqrt(4 * np.pi) * fac(2 * l + 1)))
        splines_b.append([Spline.from_data(l, rmax=r[-1], f_g=f_g)])

    assert ibzwfs.domain_comm.size == 1

    for wfs in ibzwfs:
        if wfs.spin != spin:
            continue
        psit_nX = wfs.psit_nX
        lf_blX = psit_nX.desc.atom_centered_functions(
            splines_b, spos_bc, cut=True)
        f_bnl = lf_blX.integrate(psit_nX)
        f_knB[wfs.k] = f_bnl.data
    ibzwfs.kpt_comm.sum(f_knB)
    return f_knB.conj()
