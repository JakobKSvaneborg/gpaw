from __future__ import annotations

from math import factorial as fac

import numpy as np
from ase.units import Bohr

from gpaw.core import PWArray
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from gpaw.spline import Spline
from gpaw.typing import Array2D


def _bz_wfs(ibzwfs: IBZWaveFunctions, K: int, spin: int):
    # TODO: consolidate with gpaw/response/symmetry.py (filters
    # non-symmorphic ops) and gpaw/new/pw/hybrids.py (symmorphic only).
    # This is the first code path in GPAW that handles non-symmorphic
    # symmetry for wave functions in PW space.
    """Construct BZ wave functions and PAW projections at BZ k-point *K*.

    Parameters
    ----------
    ibzwfs : IBZWaveFunctions
        Container holding IBZ wave functions.
    K : int
        Index in the full (BZ) list of k-points.
    spin : int
        Spin channel.

    Returns
    -------
    wfs : PWFDWaveFunctions
        New wave-function object whose ``psit_nX`` is a ``PWArray`` at
        the BZ k-point and whose ``P_ani`` holds the PAW projections
        recomputed at that k-point.

    Notes
    -----
    For plane-wave calculators this applies the symmetry operation
    ``{U | tau}`` (plus an optional time-reversal conjugation) that
    relates the IBZ representative to the BZ k-point.  A translation
    phase ``exp(-2 pi i (k' + G) . tau)`` is added after
    :meth:`PWArray.transform` to handle non-symmorphic operations
    (``symmorphic=False``).  PAW projections are re-integrated at the
    BZ k-point using ``atom_centered_functions``, which absorbs any
    atom permutation and Bloch-phase changes implied by the symmetry
    operation.
    """
    ibz = ibzwfs.ibz
    k = int(ibz.bz2ibz_K[K])
    wfs_ibz = ibzwfs._get_wfs(k, spin)

    s = int(ibz.s_K[K])
    time_reversal = bool(ibz.time_reversal_K[K])
    U_cc = np.asarray(ibz.symmetries.rotation_scc[s])
    tau_c = np.asarray(ibz.symmetries.translation_sc[s])

    # Fast path: identity rotation, no time reversal, no translation,
    # and the IBZ representative is this BZ k-point.  This also covers
    # non-PW modes (FD, LCAO) when no symmetry reduction is in use.
    # The ibz2bz check is defensive: it is implied by the other three
    # conditions (U=I, τ=0, no TR make k_bz = k_ibz), but guards
    # against floating-point near-misses in BZ k-point matching.
    identity = (np.array_equal(U_cc, np.eye(3, dtype=int))
                and not time_reversal
                and not tau_c.any()
                and int(ibz.ibz2bz_k[k]) == K)
    if identity:
        return wfs_ibz

    psit_nX = wfs_ibz.psit_nX
    if not isinstance(psit_nX, PWArray):
        raise NotImplementedError(
            'BZ unfolding of wave functions is only implemented for '
            'plane-wave mode.  Use mode=PW(...) or run with '
            "symmetry='off'.")

    psit_BZ = psit_nX.transform(U_cc, complex_conjugate=time_reversal)

    # Non-symmorphic correction: translation phase.
    # In GPAW's convention the real-space symmetry operation is
    #     r -> U_cc^T r - tau_c     (see Symmetries.check_positions),
    # while PWArray.transform only applies the pure-rotation part and
    # returns ψ'(r) = ψ_k(U_cc^T r) at k' = U_cc k.  The missing step
    # is the translation: the desired BZ wave function is
    #     ψ'(r - s),   with s = U_cc^{-T} tau_c.
    # That shift multiplies each plane-wave coefficient at (k', G) by
    #     exp(-2 pi i (k' + G) . s).
    # For symmorphic ops tau_c == 0, so this reduces to no-op.
    if tau_c.any():
        pw = psit_BZ.desc
        # s = U_cc^{-T} @ tau_c, solved via U_cc.T @ s = tau_c.
        s_c = np.linalg.solve(U_cc.T.astype(float), tau_c)
        phase_G = np.exp(
            -2j * np.pi
            * ((pw.kpt_c[:, None] + pw.indices_cG).T @ s_c))
        psit_BZ.data[...] = psit_BZ.data * phase_G

    # Build a new PWFDWaveFunctions holding the BZ wave functions.
    wfs_bz = PWFDWaveFunctions.from_wfs(wfs_ibz, psit_BZ)
    # Force recomputation of P_ani and pt_aiX at the new k-point.
    wfs_bz._P_ani = None
    wfs_bz._pt_aiX = None
    return wfs_bz


def get_wannier_integrals(ibzwfs: IBZWaveFunctions,
                          grid,
                          s: int,
                          K: int,
                          K1: int,
                          G_c,
                          nbands=None) -> Array2D:
    """Calculate integrals for maximally localized Wannier functions.

    *K* and *K1* are indices into the full BZ k-point list.  When no
    symmetry reduction is in use (``len(bz) == len(ibz)``) these
    coincide with IBZ indices.
    """
    ibzwfs.make_sure_wfs_are_read_from_gpw_file()
    assert s <= ibzwfs.nspins
    # XXX not for the kpoint/spin parallel case
    assert ibzwfs.comm.size == 1
    wfs = _bz_wfs(ibzwfs, K, s).to_uniform_grid_wave_functions(grid, None)
    wfs1 = _bz_wfs(ibzwfs, K1, s).to_uniform_grid_wave_functions(grid, None)
    # Get pseudo part
    psit_nR = wfs.psit_nX.data
    psit1_nR = wfs1.psit_nX.data
    Z_nn = grid._gd.wannier_matrix(psit_nR, psit1_nR, G_c, nbands)
    # Add corrections
    add_wannier_correction(Z_nn, G_c, wfs, wfs1, nbands)
    grid.comm.sum(Z_nn)
    return Z_nn


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
    """Project wave functions onto localized functions.

    Determine the projections of the Kohn-Sham eigenstates
    onto specified localized functions of the format::

      locfun = [[spos_c, l, sigma], [...]]

    spos_c can be an atom index, or a scaled position vector. l is
    the angular momentum, and sigma is the (half-) width of the
    radial gaussian.

    Return format is::

      f_Kni = <psi_Kn | f_i>

    where psi_Kn are the wave functions at BZ k-point index *K*, and
    f_i are the specified localized functions.

    As a special case, locfun can be the string 'projectors', in which
    case the bound state projectors are used as localized functions.
    """
    ibzwfs.make_sure_wfs_are_read_from_gpw_file()
    assert ibzwfs.comm.size == 1
    assert ibzwfs.domain_comm.size == 1

    ibz = ibzwfs.ibz
    nbzk = len(ibz.bz)
    relpos_ac = ibzwfs._wfs_u[0].relpos_ac

    if isinstance(locfun, str):
        assert locfun == 'projectors'
        f_Kin: list = []
        for K in range(nbzk):
            wfs = _bz_wfs(ibzwfs, K, spin)
            f_in = []
            for a, P_ni in wfs.P_ani.items():
                i = 0
                setup = wfs.setups[a]
                for ell, n in zip(setup.l_j, setup.n_j):
                    if n >= 0:
                        for j in range(i, i + 2 * ell + 1):
                            f_in.append(P_ni[:, j])
                    i += 2 * ell + 1
            f_Kin.append(f_in)
        f_Kni = np.array(f_Kin).transpose(0, 2, 1)
        return f_Kni.conj()

    nbf = int(np.sum([2 * ell + 1 for pos, ell, a in locfun]))
    f_KnB = np.zeros((nbzk, ibzwfs.nbands, nbf), ibzwfs.dtype)

    spos_bc = []
    splines_b = []
    for spos_c, ell, sigma in locfun:
        if isinstance(spos_c, int):
            spos_c = relpos_ac[spos_c]
        spos_bc.append(spos_c)
        alpha = .5 * Bohr**2 / sigma**2
        r = np.linspace(0, 10. * sigma, 500)
        f_g = (fac(ell) * (4 * alpha)**(ell + 3 / 2.) *
               np.exp(-alpha * r**2) /
               (np.sqrt(4 * np.pi) * fac(2 * ell + 1)))
        splines_b.append([Spline.from_data(ell, rmax=r[-1], f_g=f_g)])

    for K in range(nbzk):
        wfs = _bz_wfs(ibzwfs, K, spin)
        psit_nX = wfs.psit_nX
        lf_blX = psit_nX.desc.atom_centered_functions(
            splines_b, spos_bc, cut=True)
        f_bnl = lf_blX.integrate(psit_nX)
        f_KnB[K] = f_bnl.data
    return f_KnB.conj()
