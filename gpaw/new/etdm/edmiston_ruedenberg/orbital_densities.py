"""
Gamma point orbital densities in real and reciprocal spaces
"""
from math import pi

import numpy as np
from gpaw.core.atom_centered_functions import AtomArraysLayout
from gpaw.core.plane_waves import PWArray
from gpaw.new.wave_functions import WaveFunctions
from gpaw.setup import Setups
from gpaw.utilities import pack_density, unpack_hermitian


def calc_orbital_densities_real_space(psit_nX: PWArray, h: float):
    """

    Parameters
    ----------
    psit_nX : PWArray
    h : float
        grid spacing

    Returns
    -------
    rho_nR : UniformGridExpansions
    psit_nR : UniformGridExpansions
        ifft of psit_nX

    """
    grid = psit_nX.desc.uniform_grid_with_grid_spacing(h)
    # take Fourier transform of the wave-functions and
    # calculate the integral in real space
    psit_nR = grid.empty(psit_nX.dims)
    psit_nX.ifft(out=psit_nR)

    rho_nR = psit_nR.new()
    rho_nR.data[:] = psit_nR.data.conj() * psit_nR.data
    return rho_nR, psit_nR


def calc_orbital_densities_reciprocal(psit_nX: PWArray, h: float):
    """

    Parameters
    ----------
    psit_nX : PWArray
    h : float
        grid spacing

    Returns
    -------
    rho_nG : PWArray

    """
    rho_nR = calc_orbital_densities_real_space(psit_nX, h)
    rho_nG = psit_nX.new()

    for pd_G, pd_R in zip(rho_nG, rho_nR):
        pd_R.fft(out=pd_G)

    return rho_nG


def get_self_hartree_uniform(rho_nG: PWArray):
    """
    Parameters
    ----------
    rho_nG : PWArray
    Returns
    -------
    eh_n : np.ndarray
    vrho_nG : PWArray

    """

    Gsquare = rho_nG.desc.ekin_G.copy()
    if rho_nG.desc.comm.rank == 0:
        Gsquare[0] = 1
    vrho_nG = rho_nG.new()
    vrho_nG.data[:] = -pi * rho_nG.data / Gsquare

    if rho_nG.desc.comm.rank == 0:
        vrho_nG.data[:, 0] = 0

    eh_n = []
    for v_g, rho_G in zip(vrho_nG, rho_nG):
        eh_n.append(v_g.integrate(rho_G))

    return np.array(eh_n), vrho_nG


def get_self_hartree_potential(rho_nG: PWArray):
    """
    Parameters
    ----------
    rho_nG : PWArray
    Returns
    -------
    vrho_nG : PWArray

    """

    Gsquare = rho_nG.desc.ekin_G.copy()
    if rho_nG.desc.comm.rank == 0:
        Gsquare[0] = 1
    vrho_nG = rho_nG.new()
    vrho_nG.data[:] = -pi * rho_nG.data / Gsquare

    if rho_nG.desc.comm.rank == 0:
        vrho_nG.data[:, 0] = 0

    return vrho_nG


def calc_self_hartree_derivative(psit_nX: PWArray, h: float):
    """deravative of self Hartree.

    Parameters
    ----------
    psit_nX
    h

    Returns
    -------

    """
    rho_nR, psi_nR = calc_orbital_densities_real_space(psit_nX, h)

    rho_nG = psit_nX.new()
    for pd_G, pd_R in zip(rho_nG, rho_nR):
        pd_R.fft(out=pd_G)

    vh_nG = get_self_hartree_potential(rho_nG)

    vh_nR = psi_nR.new()
    vh_nG.ifft(out=vh_nR)

    vh_nR.data[:] = vh_nR.data * psi_nR.data * 2

    for pd_G, pd_R in zip(vh_nG, vh_nR):
        pd_R.fft(out=pd_G)

    return vh_nG


def orbital_compensation_charges(setups: Setups, wfs: WaveFunctions):
    """Calculate compensation charges

    Parameters
    ----------
    setups : Setups
    wfs : WaveFunctions

    Returns
    -------
    Q_aNL : AtomArraysLayout
        Compensation charges for pair-densities

    """

    P_ani = wfs.P_ani
    n = P_ani.dims[0]

    Q_anL = AtomArraysLayout(
        [(n, setup.Delta_iiL.shape[2]) for setup in setups],
        atomdist=P_ani.layout.atomdist,
        xp=wfs.xp,
    ).empty()

    for a, P_ni in P_ani.items():
        Q_anL[a] = np.einsum(
            "ni,nj,ijL->nnL", P_ni.conj(), P_ni, setups[a].Delta_iiL, optimize=True
        )

    return Q_anL


def self_hartree_paw(
    wfs: WaveFunctions,
    setups: Setups,
    ghat_aLG,
    vHt_nG,
    domian_sum: bool = True,
):
    r"""Calculate

    Parameters
    ----------
    wfs : WaveFunctions
        wfs is provided for the use of projections
    setups : Setups
    domian_sum :  bool
        Sum the result over different domains if run in parallel

    Returns
    -------
    Returns
    -------
    v_n : np.ndarray
    dG_nm : np.ndarray
    """

    P_ani = wfs.P_ani
    n = wfs.P_ani.dims[0]

    dH_anp = AtomArraysLayout(
        [(n, setup.M_pp.shape[0]) for setup in setups],
        atomdist=P_ani.layout.atomdist,
        xp=wfs.xp,
        dtype=wfs.dtype,
    ).zeros()

    # W_anL = AtomArraysLayout(
    #     [(n, setup.Delta_iiL.shape[2]) for setup in setups],
    #     atomdist=P_ani.layout.atomdist,
    #     xp=wfs.xp,
    # ).empty()

    # W_anL = ghat_aLG.integrate(vHt_nG)

    for a, P_ni in P_ani.items():
        M_pp = setups[a].M_pp
        D_nij = P_ni[:, np.newaxis, :].conj() * P_ni[:, :, np.newaxis]
        D_np = np.array([pack_density(D_ij) for D_ij in D_nij])
        M_np = D_np @ M_pp

        dH_anp[a] += 2.0 * M_np  # + W_anL[a] @ setups[a].Delta_pL.T

    dG_nm = np.zeros(shape=(n, n), dtype=wfs.dtype)
    for a, P_ni in P_ani.items():
        dH_nii = np.array([unpack_hermitian(dH_p) for dH_p in dH_anp[a]])
        dG_nm -= np.einsum("ni,mij,mj->nm", P_ni.conj(), dH_nii, P_ni, optimize=True)

    if domian_sum:
        wfs.domain_comm.sum(dG_nm)
    return dG_nm
