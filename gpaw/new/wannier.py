from __future__ import annotations

from math import factorial as fac

import numpy as np
from ase.units import Bohr

from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from gpaw.spline import Spline
from gpaw.typing import Array2D


def _get_bz_wfs(ibzwfs: IBZWaveFunctions, K: int, spin: int, grid):
    """Get uniform-grid wave functions at BZ k-point K.

    When symmetry reduction is used, IBZ wave functions are
    symmetry-transformed to the requested BZ k-point on the fly.

    Parameters
    ----------
    ibzwfs : IBZWaveFunctions
        Wave functions stored for IBZ k-points.
    K : int
        BZ k-point index.
    spin : int
        Spin channel.
    grid
        Uniform grid descriptor.

    Returns
    -------
    PWFDWaveFunctions
        Wave functions on the uniform real-space grid at BZ k-point K.
    """
    ibz = ibzwfs.ibz
    k = ibz.bz2ibz_K[K]          # IBZ k-point index
    s_op = ibz.s_K[K]            # symmetry operation index
    tr = ibz.time_reversal_K[K]  # time reversal flag

    wfs_ibz = ibzwfs._get_wfs(k, spin)

    # Fast path: no transformation needed for IBZ representatives
    if s_op == 0 and not tr:
        return wfs_ibz.to_uniform_grid_wave_functions(grid, None)

    # Transform PW coefficients from IBZ to BZ k-point.
    # This requires plane-wave representation (not available in FD mode).
    if not hasattr(wfs_ibz.psit_nX, 'transform'):
        raise NotImplementedError(
            'K-point symmetry unfolding requires plane-wave (PW) mode.  '
            "Use mode='pw' or set symmetry='off' for FD/LCAO mode.")
    U_cc = ibz.symmetries.rotation_scc[s_op]
    psit_nG_bz = wfs_ibz.psit_nX.transform(U_cc, tr)

    # Build new wave functions object with transformed PW coefficients.
    # PAW projections (P_ani) are recomputed lazily from the new
    # PW coefficients when accessed.
    wfs_bz = PWFDWaveFunctions.from_wfs(wfs_ibz, psit_nG_bz)

    # Convert to uniform grid (FFT to real space)
    return wfs_bz.to_uniform_grid_wave_functions(grid, None)


def get_wannier_integrals(ibzwfs: IBZWaveFunctions,
                          grid,
                          s: int,
                          K: int,
                          K1: int,
                          G_c,
                          nbands=None) -> Array2D:
    """Calculate integrals for maximally localized Wannier functions.

    K and K1 are BZ k-point indices.  When symmetry reduction is
    active, wave functions are obtained by symmetry-transforming
    the IBZ wave functions.
    """
    ibzwfs.make_sure_wfs_are_read_from_gpw_file()
    assert s <= ibzwfs.nspins
    # XXX not for the kpoint/spin parallel case
    assert ibzwfs.comm.size == 1
    wfs = _get_bz_wfs(ibzwfs, K, s, grid)
    wfs1 = _get_bz_wfs(ibzwfs, K1, s, grid)
    # Get pseudo part
    psit_nR = wfs.psit_nX.data
    psit1_nR = wfs1.psit_nX.data
    Z_nn = grid._gd.wannier_matrix(psit_nR, psit1_nR, G_c, nbands)
    # Add corrections
    add_wannier_correction(Z_nn, G_c, wfs, wfs1, nbands)
    grid.comm.sum(Z_nn)
    return Z_nn


def get_bz_pseudo_wave_function(ibzwfs: IBZWaveFunctions,
                                grid,
                                band: int,
                                K: int,
                                spin: int) -> np.ndarray:
    """Get pseudo wave function on a real-space grid at a BZ k-point.

    Unlike the standard ``get_pseudo_wave_function`` (which uses IBZ
    indices), this accepts BZ k-point indices and handles
    symmetry-unfolding transparently.

    Parameters
    ----------
    ibzwfs : IBZWaveFunctions
        Wave functions stored for IBZ k-points.
    grid
        Uniform grid descriptor.
    band : int
        Band index.
    K : int
        BZ k-point index.
    spin : int
        Spin channel.

    Returns
    -------
    ndarray
        Pseudo wave function on the real-space grid.
    """
    ibzwfs.make_sure_wfs_are_read_from_gpw_file()
    wfs = _get_bz_wfs(ibzwfs, K, spin, grid)
    return wfs.psit_nX.data[band]


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

      f_kni = <psi_kn | f_i>

    where psi_kn are the wave functions, and f_i are the specified
    localized functions.

    As a special case, locfun can be the string 'projectors', in which
    case the bound state projectors are used as localized functions.

    When symmetry reduction is active (IBZ != BZ), wave functions at
    non-IBZ k-points are obtained by symmetry transformation.  The
    returned projections cover all BZ k-points.
    """
    ibz = ibzwfs.ibz
    nbz = len(ibz.bz)
    nibz = len(ibz)
    symmetry_active = (nibz < nbz)

    if isinstance(locfun, str):
        assert locfun == 'projectors'
        if not symmetry_active:
            # Original path: iterate IBZ = BZ
            f_kin = []
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
                    f_kin.append(f_in)
            f_kni = np.array(f_kin).transpose(0, 2, 1)
            return f_kni.conj()
        else:
            # Symmetry-aware path: get projections at all BZ k-points
            # by using _get_bz_wfs for the 'projectors' case.
            # We need a grid descriptor for _get_bz_wfs.
            # For projectors, we can get P_ani from PW wave functions
            # without converting to uniform grid, but _get_bz_wfs
            # returns uniform-grid wfs that have P_ani.
            raise NotImplementedError(
                "The 'projectors' initial guess is not yet supported "
                "with k-point symmetry reduction.  Use 'bloch', "
                "'random', or 'orbitals' instead.")

    nkpts = nbz if symmetry_active else nibz
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

    if not symmetry_active:
        # Original path
        for wfs in ibzwfs:
            if wfs.spin != spin:
                continue
            psit_nX = wfs.psit_nX
            lf_blX = psit_nX.desc.atom_centered_functions(
                splines_b, spos_bc, cut=True)
            f_bnl = lf_blX.integrate(psit_nX)
            f_knB[wfs.q] = f_bnl.data
    else:
        # Symmetry-aware path: project at all BZ k-points.
        # For each BZ k-point K, get wave functions (possibly
        # symmetry-transformed) and compute projections.
        for K in range(nbz):
            k = ibz.bz2ibz_K[K]
            s_op = ibz.s_K[K]
            tr = ibz.time_reversal_K[K]

            wfs_ibz = ibzwfs._get_wfs(k, spin)

            if s_op == 0 and not tr:
                psit_nX = wfs_ibz.psit_nX
            else:
                U_cc = ibz.symmetries.rotation_scc[s_op]
                psit_nX = wfs_ibz.psit_nX.transform(U_cc, tr)

            lf_blX = psit_nX.desc.atom_centered_functions(
                splines_b, spos_bc, cut=True)
            f_bnl = lf_blX.integrate(psit_nX)
            f_knB[K] = f_bnl.data

    return f_knB.conj()
