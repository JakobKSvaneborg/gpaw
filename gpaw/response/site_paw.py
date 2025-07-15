import numpy as np

from gpaw.gaunt import gaunt
from gpaw.hubbard import aoom
from gpaw.sphere.rshe import RealSphericalHarmonicsExpansion
from gpaw.sphere.integrate import radial_truncation_function
from gpaw.utilities import unpack_density


def calculate_site_matrix_element_correction(
        pawdata, rshe: RealSphericalHarmonicsExpansion,
        rcut_p, drcut, lambd_p):
    r"""Calculate the PAW correction to a site matrix element.

    For the site matrix element

    f^ap_nn' = <ψ_n|Θ(r∊Ω_ap)f(r)|ψ_n'>

    the PAW correction for each pair of partial waves i and i' is given by
               l
           __  __
     ap    \   \   m,mi,mi' /                        a    a     ̰ a   ̰ a
    F    = /   /  g         | r^2 dr θ(r<rc) f (r) [φ(r) φ(r) - φ(r) φ(r)]
     ii'   ‾‾  ‾‾  l,li,li' /         p       lm     i    i'     i    i'
           l  m=-l

    where g refer to the Gaunt coefficients and f_lm(r) are the real
    spherical harmonics expansion coefficients of the function f(r).

    Here, we evaluate the correction F_ii'^ap for various smooth truncation
    functions θ_p(r<rc), parametrized by rc, Δrc and λ.
    """
    rgd = rshe.rgd
    assert rgd is pawdata.xc_correction.rgd
    ni = pawdata.ni  # Number of partial waves
    l_j = pawdata.l_j  # l-index for each radial function index j
    lmax = max(l_j)
    G_LLL = gaunt(lmax)
    assert max(rshe.l_M) <= lmax * 2
    # (Real) radial functions for the partial waves
    phi_jg = pawdata.phi_jg
    phit_jg = pawdata.phit_jg
    # Truncate the radial functions to span only the radial grid coordinates
    # which need correction
    assert np.allclose(rgd.r_g - pawdata.rgd.r_g[:rgd.N], 0.)
    phi_jg = np.array(phi_jg)[:, :rgd.N]
    phit_jg = np.array(phit_jg)[:, :rgd.N]

    # Calculate smooth truncation functions and allocate array
    Np = len(rcut_p)
    assert len(lambd_p) == Np
    theta_pg = [radial_truncation_function(rgd.r_g, rcut, drcut, lambd)
                for rcut, lambd in zip(rcut_p, lambd_p)]
    F_pii = np.zeros((Np, ni, ni), dtype=float)

    # Loop of radial function indices for partial waves i and i'
    i1_counter = 0
    for j1, l1 in enumerate(l_j):
        i2_counter = 0
        for j2, l2 in enumerate(l_j):
            # Calculate the radial partial wave correction
            dn_g = phi_jg[j1] * phi_jg[j2] - phit_jg[j1] * phit_jg[j2]

            # Generate m-indices for each radial function
            for m1 in range(2 * l1 + 1):
                for m2 in range(2 * l2 + 1):
                    # Set up the i=(l,m) index for each partial wave
                    i1 = i1_counter + m1
                    i2 = i2_counter + m2

                    # Loop through the real spherical harmonics of the local
                    # function f(r)
                    for L, f_g in zip(rshe.L_M, rshe.f_gM.T):
                        # Angular integral
                        gaunt_coeff = G_LLL[l1**2 + m1, l2**2 + m2, L]
                        if gaunt_coeff == 0:
                            continue
                        # Radial integral
                        for p, theta_g in enumerate(theta_pg):
                            F_pii[p, i1, i2] += \
                                gaunt_coeff * rgd.integrate_trapz(
                                theta_g * f_g * dn_g)

            # Add to i and i' counters
            i2_counter += 2 * l2 + 1
        i1_counter += 2 * l1 + 1
    return F_pii


def calculate_nonlocal_hubbard_potential(D_sp, pawdata):
    r"""Calculate the Hubbard correction to the spin pair energy.

    The Hubbard correction to the site spin pair energy is given by

                     U^a
    Δd^(xc,a)_nn' = -‾‾‾  <ψ_n|m^a|ψ_n'>,
                      2

    where m^a is the nonlocal partial wave magnetization magnitude of the
    Hubbard corrected orbital subspace. To compute the correction we therefore
    need the nonlocal magnetic Hubbard potential

     U,a       U^a /  a,↑↑    a,↓↓ \ _   _a
    W      = - ‾‾‾ | ρ     - ρ     | e ⋅ u
     z,ii'      2  \  ii'     ii'  /  z

    for each pair of partial waves i and i' in the subspace.
    """
    # Extract the Hubbard corrected angular momentum and U-value (we allow only
    # Hubbard corrections of a single angular momentum per atom) along with the
    # applied Hubbard scaling
    assert pawdata.hubbard_u is not None
    assert len(pawdata.hubbard_u.l) == 1
    assert len(pawdata.hubbard_u.U) == 1
    hubbardl = pawdata.hubbard_u.l[0]
    hubbardU = pawdata.hubbard_u.U[0]
    scale = pawdata.hubbard_u.scale[0]

    # Extract data about the angular part of the partial waves
    ni = pawdata.ni  # Number of partial waves
    l_j = pawdata.l_j  # l-index for each radial function index j
    nm_j = 2 * np.array(l_j) + 1  # number of m-indices for each l(j)

    # Unpack the density matrix and allocate the output array
    D_sii = unpack_density(D_sp)
    assert D_sii.shape[1:] == (ni, ni)
    WzU_ii = np.zeros_like(D_sii[0])

    # Get atomic orbital occupation matrices (aoom) and calculate the nonlocal
    # magnetization matrix in the Hubbard corrected subspace (m-indices for the
    # given angular momentum l)
    N0_mm, _ = aoom(
        D_sii[0], hubbardl, l_j, pawdata.n_j, pawdata.N0_q, scale)
    N1_mm, dHU_ii = aoom(
        D_sii[1], hubbardl, l_j, pawdata.n_j, pawdata.N0_q, scale)
    M_mm = N0_mm.T - N1_mm.T  # ρ^(↑↑) - ρ^(↓↓)

    # Loop over radial function indices for partial waves i and i' and map each
    # l-specific (m,m') subspace to the global partial wave indices (i,i')
    for j1, l1 in enumerate(l_j):
        i1_m = slice(nm_j[:j1].sum(), nm_j[:(j1 + 1)].sum())
        for j2, l2 in enumerate(l_j):
            i2_m = slice(nm_j[:j2].sum(), nm_j[:(j2 + 1)].sum())
            if not (l1 == l2 == hubbardl):
                continue  # no correction
            # Apply scaling to appropriately account for the norm of
            # bounded/unbounded radial functions
            m_mm = M_mm * dHU_ii[i1_m, i2_m]
            # Multiply with (e_z ⋅ u) to get the nonlocal magnetization
            # *magnitude*
            eig_n, _ = np.linalg.eigh(m_mm)
            if np.max(eig_n) < 0:
                m_mm *= -1
            # Add nonlocal Hubbard potential to the output
            WzU_ii[i1_m, i2_m] = -hubbardU * m_mm / 2
    return WzU_ii
