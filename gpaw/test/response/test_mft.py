"""Calculate the Heisenberg exchange constants in Fe and Co using the MFT.
Test with unrealisticly loose parameters to catch if the numerics change.
"""

# General modules
from abc import abstractmethod

import pytest
import numpy as np

# Script modules
from ase.units import Ha

from gpaw import GPAW

from gpaw.response import ResponseGroundStateAdapter, ResponseContext
from gpaw.response.chiks import ChiKSCalculator, smat
from gpaw.response.localft import LocalFTCalculator, LocalPAWFTCalculator
from gpaw.response.site_data import AtomicSites, AtomicSiteData
from gpaw.response.mft import (IsotropicExchangeCalculator,
                               StaticSitePairFunction,
                               TwoParticleSiteMagnetizationCalculator,
                               TwoParticleSiteSpinSplittingCalculator)
from gpaw.response.site_kernels import (SphericalSiteKernels,
                                        CylindricalSiteKernels,
                                        ParallelepipedicSiteKernels)
from gpaw.response.heisenberg import (calculate_single_site_magnon_energies,
                                      calculate_fm_magnon_energies)
from gpaw.response.pair_integrator import PairFunctionIntegrator
from gpaw.response.pair_transitions import PairTransitions
from gpaw.response.matrix_elements import (SitePairDensityCalculator,
                                           SitePairSpinSplittingCalculator)
from gpaw.test.conftest import response_band_cutoff
from gpaw.test.response.test_chiks import generate_qrel_q, get_q_c


@pytest.mark.response
def test_Fe_bcc(in_tmp_dir, gpw_files):
    # ---------- Inputs ---------- #

    # MFT calculation
    ecut = 50
    # Do the high symmetry points of the bcc lattice
    q_qc = np.array([[0, 0, 0],           # Gamma
                     [0.5, -0.5, 0.5],    # H
                     [0.0, 0.0, 0.5],     # N
                     ])
    # Define site kernels to test
    # Test a single site of spherical and cylindrical geometries
    rc_pa = np.array([[1.0], [1.5], [2.0]])
    hc_pa = np.array([[1.0], [1.5], [2.0]])
    ez_pav = np.array([[[1., 0., 0.]], [[0., 1., 0.]], [[0., 0., 1.]]])

    # ---------- Script ---------- #

    # Extract the ground state fixture
    calc = GPAW(gpw_files['fe_pw'], parallel=dict(domain=1))
    nbands = response_band_cutoff['fe_pw']
    atoms = calc.atoms

    # Set up site kernels with a single site
    positions = atoms.get_positions()
    sitekernels = SphericalSiteKernels(positions, rc_pa)
    sitekernels.append(CylindricalSiteKernels(positions, ez_pav,
                                              rc_pa, hc_pa))
    # Set up a kernel to fill out the entire unit cell
    sitekernels.append(ParallelepipedicSiteKernels(positions,
                                                   [[atoms.get_cell()]]))

    # Initialize the Heisenberg exchange calculator
    gs = ResponseGroundStateAdapter(calc)
    context = ResponseContext()
    chiks_calc = ChiKSCalculator(gs, context,
                                 ecut=ecut, nbands=nbands, gammacentered=True)
    localft_calc = LocalFTCalculator.from_rshe_parameters(gs, context)
    isoexch_calc = IsotropicExchangeCalculator(chiks_calc, localft_calc)

    # Allocate array for the exchange constants
    nq = len(q_qc)
    nsites = sitekernels.nsites
    npartitions = sitekernels.npartitions
    J_qabp = np.empty((nq, nsites, nsites, npartitions), dtype=complex)

    # Calcualate the exchange constant for each q-point
    for q, q_c in enumerate(q_qc):
        J_qabp[q] = isoexch_calc(q_c, sitekernels)

    # Since we only have a single site, reduce the array
    J_qp = J_qabp[:, 0, 0, :]

    # Calculate the magnon energies
    mm = 2.21
    mm_ap = mm * np.ones((1, npartitions))  # Magnetic moments
    mw_qp = calculate_fm_magnon_energies(J_qabp, q_qc, mm_ap)[:, 0, :]

    # Compare results to test values
    test_J_pq = np.array(
        [[2.1907596825086455, 1.172424411323134, 1.6060583789867644],
         [2.612428039019977, 1.2193926800088601, 1.7635196888465006],
         [6.782367391186284, 0.2993922109834177, 1.9346016211386057],
         [1.5764800860123762, 0.8365204592352894, 1.1648584638500161],
         [2.4230224513213234, 1.2179759558303274, 1.6691805687218078],
         [5.35668502504496, 0.3801778545994659, 1.6948968244858478],
         [2.523580017606111, 1.21779750159267, 1.7637120466695273]])
    test_mw_pq = np.array(
        [[0.0, 0.9215703811633589, 0.5291414511510236],
         [0.0, 1.2606654832679791, 0.7682428508357253],
         [0.0, 5.866945864436984, 4.38711834393455],
         [0.0, 0.6696467210652369, 0.3725082553505521],
         [0.0, 1.0905398149239784, 0.682209848506349],
         [0.0, 4.503626398593207, 3.313835475619106],
         [0.0, 1.181703634401304, 0.6876633221145555]])

    # Exchange constants
    assert J_qp.imag == pytest.approx(0.0)
    assert J_qp.T.real == pytest.approx(test_J_pq, rel=2e-3)

    # Magnon energies
    assert mw_qp.T == pytest.approx(test_mw_pq, rel=2e-3)


@pytest.mark.response
def test_Co_hcp(in_tmp_dir, gpw_files):
    # ---------- Inputs ---------- #

    # MFT calculation
    ecut = 100
    # Do high symmetry points of the hcp lattice
    q_qc = np.array([[0, 0, 0],              # Gamma
                     [0.5, 0., 0.],          # M
                     [0., 0., 0.5]           # A
                     ])

    # Use spherical site kernels in a radius range which should yield
    # stable results
    rc_pa = np.array([[1.0, 1.0], [1.1, 1.1], [1.2, 1.2]])

    # Unfortunately, the usage of symmetry leads to such extensive repetition
    # of random noise, that one cannot trust individual values of J very well.
    # This is improved when increasing the number of k-points, but the problem
    # never completely vanishes
    J_atol = 1.e-2
    J_rtol = 5.e-2
    # However, derived physical values have an increased error cancellation due
    # to their collective nature.
    mw_rtol = 25e-3  # relative tolerance of absolute results
    mw_ctol = 5.e-2  # relative tolerance on kernel and eta self-consistency

    # ---------- Script ---------- #

    # Extract the ground state fixture
    calc = GPAW(gpw_files['co_pw'], parallel=dict(domain=1))
    nbands = response_band_cutoff['co_pw']
    atoms = calc.get_atoms()

    # Set up spherical site kernels
    positions = atoms.get_positions()
    sitekernels = SphericalSiteKernels(positions, rc_pa)

    # Set up a site kernel to fill out the entire unit cell
    cell_cv = atoms.get_cell()
    cc_v = np.sum(cell_cv, axis=0) / 2.  # Unit cell center
    ucsitekernels = ParallelepipedicSiteKernels([cc_v], [[cell_cv]])

    # Initialize the exchange calculator with and without symmetry
    gs = ResponseGroundStateAdapter(calc)
    context = ResponseContext()
    chiks_calc0 = ChiKSCalculator(gs, context,
                                  disable_point_group=True,
                                  disable_time_reversal=True,
                                  ecut=ecut, nbands=nbands, gammacentered=True)
    localft_calc = LocalPAWFTCalculator(gs, context)
    isoexch_calc0 = IsotropicExchangeCalculator(chiks_calc0, localft_calc)
    chiks_calc1 = ChiKSCalculator(gs, context,
                                  ecut=ecut, nbands=nbands, gammacentered=True)
    isoexch_calc1 = IsotropicExchangeCalculator(chiks_calc1, localft_calc)

    # Allocate array for the spherical site exchange constants
    nq = len(q_qc)
    nsites = sitekernels.nsites
    npartitions = sitekernels.npartitions
    J_qabp = np.empty((nq, nsites, nsites, npartitions), dtype=complex)

    # Allocate array for the unit cell site exchange constants
    Juc_qs = np.empty((nq, 2), dtype=complex)

    # Calcualate the exchange constants for each q-point
    for q, q_c in enumerate(q_qc):
        J_qabp[q] = isoexch_calc0(q_c, sitekernels)
        chiksr_buffer = isoexch_calc0._chiksr
        Juc_qs[q, 0] = isoexch_calc0(q_c, ucsitekernels)[0, 0, 0]
        assert isoexch_calc0._chiksr is chiksr_buffer, \
            'Two subsequent IsotropicExchangeCalculator calls with the same '\
            'q_c, should reuse, not update, the chiks buffer'

        Juc_qs[q, 1] = isoexch_calc1(q_c, ucsitekernels)[0, 0, 0]

    # Calculate the magnon energy
    mom = atoms.get_magnetic_moment()
    mm_ap = mom / 2.0 * np.ones((nsites, npartitions))
    mw_qnp = calculate_fm_magnon_energies(J_qabp, q_qc, mm_ap)
    mw_qnp = np.sort(mw_qnp, axis=1)  # Make sure the eigenvalues are sorted
    mwuc_qs = calculate_single_site_magnon_energies(Juc_qs, q_qc, mom)

    # Compare results to test values
    print(J_qabp[..., 1], mw_qnp[..., 1], mwuc_qs[:, 0])
    test_J_qab = np.array([[[1.23106207 - 0.j, 0.25816335 - 0.j],
                            [0.25816335 + 0.j, 1.23106207 + 0.j]],
                           [[0.88823839 + 0.j, 0.07345416 - 0.04947835j],
                            [0.07345416 + 0.04947835j, 0.88823839 + 0.j]],
                           [[1.09349955 - 0.j, 0.00000010 - 0.01176761j],
                            [0.00000010 + 0.01176761j, 1.09349955 - 0.j]]])
    test_mw_qn = np.array([[0., 0.64793939],
                           [0.64304039, 0.86531921],
                           [0.48182997, 0.51136436]])
    test_mwuc_q = np.array([0., 0.69678659, 0.44825874])

    # Exchange constants
    assert J_qabp[..., 1] == pytest.approx(test_J_qab, abs=J_atol, rel=J_rtol)

    # Magnon energies
    assert np.all(np.abs(mw_qnp[0, 0, :]) < 1.e-8)  # Goldstone theorem
    assert np.allclose(mwuc_qs[0, :], 0.)  # Goldstone
    assert mw_qnp[1:, 0, 1] == pytest.approx(test_mw_qn[1:, 0], rel=mw_rtol)
    assert mw_qnp[:, 1, 1] == pytest.approx(test_mw_qn[:, 1], rel=mw_rtol)
    assert mwuc_qs[1:, 0] == pytest.approx(test_mwuc_q[1:], rel=mw_rtol)

    # Check self-consistency of results
    # We should be in a radius range, where the magnon energies don't change
    assert np.allclose(mw_qnp[1:, 0, ::2],
                       test_mw_qn[1:, 0, np.newaxis], rtol=mw_ctol)
    assert np.allclose(mw_qnp[:, 1, ::2],
                       test_mw_qn[:, 1, np.newaxis], rtol=mw_ctol)
    # Check that symmetry toggle do not change the magnon energies
    assert np.allclose(mwuc_qs[1:, 0], mwuc_qs[1:, 1], rtol=mw_ctol)


@pytest.mark.response
@pytest.mark.parametrize('qrel', generate_qrel_q())
def test_Co_site_magnetization_sum_rule(in_tmp_dir, gpw_files, qrel):
    # Set up ground state adapter and atomic site data
    calc = GPAW(gpw_files['co_pw'], parallel=dict(domain=1))
    nbands = response_band_cutoff['co_pw']
    gs = ResponseGroundStateAdapter(calc)
    context = ResponseContext('Co_sum_rule.txt')
    site_data = get_co_atomic_site_data(gs)
    nblocks = generate_nblocks(context)

    # Get wave vector to test
    q_c = get_q_c('co_pw', qrel)

    # ----- Single-particle site magnetization ----- #
    # Set up calculator and calculate the site magnetization
    simple_site_mag_calc = SingleParticleSiteMagnetizationCalculator(
        gs, site_data.sites, context)
    ssite_mag_ar = simple_site_mag_calc()

    # Test that the imaginary part vanishes (we use only diagonal pair
    # densities correcsponding to |ψ_nks(r)|^2)
    assert np.allclose(ssite_mag_ar.imag, 0.)
    ssite_mag_ar = ssite_mag_ar.real

    # Test that the results match a conventional calculation
    magmom_ar = site_data.calculate_magnetic_moments()
    assert ssite_mag_ar == pytest.approx(magmom_ar, rel=5e-3)

    # ----- Two-particle site magnetization ----- #
    # Set up calculator and calculate site magnetization by sum rule
    sum_rule_site_mag_calc = TwoParticleSiteMagnetizationCalculator(
        gs, site_data.sites, context, nblocks=nblocks, nbands=nbands)
    site_mag_abr = sum_rule_site_mag_calc(q_c)
    context.write_timer()

    # Test that the sum rule site magnetization is a positive-valued diagonal
    # real array
    site_mag_ra = site_mag_abr.diagonal()
    assert np.all(site_mag_ra.real > 0)
    assert np.all(np.abs(site_mag_ra.imag) / site_mag_ra.real < 1e-6)
    site_mag_ra = site_mag_ra.real
    assert np.all(np.abs(np.diagonal(np.fliplr(  # off-diagonal elements
        site_mag_abr))) / site_mag_ra < 5e-2)
    site_mag_ar = site_mag_ra.T
    # Test that the magnetic moments on the two Co atoms are identical
    assert site_mag_ar[0] == pytest.approx(site_mag_ar[1], rel=1e-4)

    # Test that the result more or less matches a conventional calculation at
    # close-packing
    assert np.average(site_mag_ar, axis=0)[-1] == pytest.approx(
        np.average(magmom_ar, axis=0)[-1], rel=5e-2)

    # Test values against reference
    print(np.average(site_mag_ar, axis=0)[::2])
    assert np.average(site_mag_ar, axis=0)[::2] == pytest.approx(
        np.array([3.91823444e-04, 1.45641911e-01, 6.85939109e-01,
                  1.18813171e+00, 1.49761591e+00, 1.58954270e+00]), rel=5e-2)

    # import matplotlib.pyplot as plt
    # rc_r = site_data.sites.rc_ap[0] * Bohr
    # plt.plot(rc_r, site_mag_ar[0], '-o', mec='k')
    # plt.plot(rc_r, ssite_mag_ar[0], '-o', mec='k', zorder=1)
    # plt.plot(rc_r, magmom_ar[0], '-o', mec='k', zorder=0)
    # plt.xlabel(r'$r_\mathrm{c}$ [$\mathrm{\AA}$]')
    # plt.ylabel(r'$m$ [$\mu_\mathrm{B}$]')
    # plt.title(str(q_c))
    # plt.show()


@pytest.mark.response
@pytest.mark.parametrize('qrel', generate_qrel_q())
def test_Co_site_spin_splitting_sum_rule(in_tmp_dir, gpw_files, qrel):
    # Set up ground state adapter and atomic site data
    calc = GPAW(gpw_files['co_pw'], parallel=dict(domain=1))
    nbands = response_band_cutoff['co_pw']
    gs = ResponseGroundStateAdapter(calc)
    context = ResponseContext('Co_sum_rule.txt')
    site_data = get_co_atomic_site_data(gs)
    nblocks = generate_nblocks(context)

    # Get wave vector to test
    q_c = get_q_c('co_pw', qrel)

    # ----- Single-particle site spin splitting ----- #
    # Set up calculator and calculate the site magnetization
    single_particle_dxc_calc = SingleParticleSiteSpinSplittingCalculator(
        gs, site_data.sites, context)
    single_particle_dxc_ar = single_particle_dxc_calc()

    # Test that the imaginary part vanishes (we use only diagonal pair
    # spin splitting densities correcsponding to -2W_xc^z(r)|ψ_nks(r)|^2)
    assert np.allclose(single_particle_dxc_ar.imag, 0.)
    single_particle_dxc_ar = single_particle_dxc_ar.real

    # Test that the results match a conventional calculation
    dxc_ar = site_data.calculate_spin_splitting()
    assert single_particle_dxc_ar == pytest.approx(dxc_ar, rel=5e-3)

    # ----- Two-particle site spin splitting ----- #
    # Set up calculator and calculate site spin splitting by sum rule
    two_particle_dxc_calc = TwoParticleSiteSpinSplittingCalculator(
        gs, site_data.sites, context, nblocks=nblocks, nbands=nbands)
    tp_dxc_abr = two_particle_dxc_calc(q_c)
    context.write_timer()

    # Test that the two-particle spin splitting is a positive-valued diagonal
    # real array
    tp_dxc_ra = tp_dxc_abr.diagonal()
    assert np.all(tp_dxc_ra.real > 0)
    assert np.all(np.abs(tp_dxc_ra.imag) / tp_dxc_ra.real < 1e-4)
    tp_dxc_ra = tp_dxc_ra.real
    assert np.all(np.abs(np.diagonal(np.fliplr(  # off-diagonal elements
        tp_dxc_abr))) / tp_dxc_ra < 5e-2)
    tp_dxc_ar = tp_dxc_ra.T
    # Test that the spin splitting on the two Co atoms is identical
    assert tp_dxc_ar[0] == pytest.approx(tp_dxc_ar[1], rel=1e-4)

    # Test values against reference
    print(np.average(tp_dxc_ar, axis=0)[::2])
    assert np.average(tp_dxc_ar, axis=0)[::2] == pytest.approx(
        np.array([3.68344584e-04, 3.13780575e-01, 1.35409600e+00,
                  2.14237563e+00, 2.52032513e+00, 2.61406726e+00]), rel=5e-2)

    # import matplotlib.pyplot as plt
    # rc_r = site_data.sites.rc_ap[0] * Bohr
    # plt.plot(rc_r, tp_dxc_ar[0], '-o', mec='k')
    # plt.plot(rc_r, single_particle_dxc_ar[0], '-o', mec='k', zorder=1)
    # plt.plot(rc_r, dxc_ar[0], '-o', mec='k', zorder=0)
    # plt.xlabel(r'$r_\mathrm{c}$ [$\mathrm{\AA}$]')
    # plt.ylabel(r'$\Delta_\mathrm{xc}$ [eV]')
    # plt.title(str(q_c))
    # plt.show()


# ---------- Test functionality ---------- #


def get_co_atomic_site_data(gs):
    # Set up atomic sites
    rmin_a, _ = AtomicSiteData.valid_site_radii_range(gs)
    # Make sure that the two sites do not overlap
    nn_dist = min(2.5071, np.sqrt(2.5071**2 / 3 + 4.0695**2 / 4))
    rc_r = np.linspace(rmin_a[0], nn_dist / 2, 11)
    sites = AtomicSites(indices=[0, 1], radii=[rc_r, rc_r])
    return AtomicSiteData(gs, sites)


def generate_nblocks(context):
    if context.comm.size % 4 == 0:
        nblocks = 4
    elif context.comm.size % 2 == 0:
        nblocks = 2
    else:
        nblocks = 1
    return nblocks


class SingleParticleSiteQuantity(StaticSitePairFunction):
    @property
    def shape(self):
        return self.sites.shape


class SingleParticleSiteSumRuleCalculator(PairFunctionIntegrator):
    r"""Calculator for single-particle site sum rules.

    For any site matrix element f^a_(nks,n'k's'), one may define a single-
    particle site sum rule by considering only the diagonal of the matrix
    element:
                 __  __
             1   \   \
    f_a^μ = ‾‾‾  /   /  σ^μ_ss f_nks f^a_(nks,nks)
            N_k  ‾‾  ‾‾
                 k   n,s

    where μ∊{0,z}.
    """

    def __init__(self, gs, sites, context):
        super().__init__(gs, context,
                         disable_point_group=True,
                         disable_time_reversal=True)

        # Set up calculator for the f^a matrix element
        self.sites = sites
        self.matrix_element_calc = self.create_matrix_element_calculator()

    @abstractmethod
    def create_matrix_element_calculator(self):
        """Create the desired site matrix element calculator."""

    def __call__(self):
        # Set up transitions
        # Loop over bands, which are fully or partially occupied
        nocc2 = self.kptpair_extractor.nocc2
        n_n = list(range(nocc2))
        n_t = np.array(n_n + n_n)
        s_t = np.array([0] * nocc2 + [1] * nocc2)
        transitions = PairTransitions(n1_t=n_t, n2_t=n_t, s1_t=s_t, s2_t=s_t)

        # Set up data object with q=0
        qpd = self.get_pw_descriptor([0., 0., 0.], ecut=1e-3)
        site_quantity = SingleParticleSiteQuantity(qpd, self.sites)

        # Perform actual calculation
        self._integrate(site_quantity, transitions)

        return site_quantity.array

    def add_integrand(self, kptpair, weight, site_quantity):
        r"""Add the integrand of the outer k-point integral.

        With
                   __
                1  \
        f_a^μ = ‾  /  (...)_k
                V  ‾‾
                   k

        the integrand has to be multiplied with the cell volume V0:
                     __
                     \
        (...)_k = V0 /  σ^μ_ss f_nks f^a_(nks,nks)
                     ‾‾
                     n,s
        """
        # Calculate matrix elements
        site_matrix_element = self.matrix_element_calc(
            kptpair, site_quantity.qpd)
        assert site_matrix_element.tblocks.blockcomm.size == 1
        f_tap = site_matrix_element.get_global_array()

        # Calculate Pauli matrix factors and multiply the occupations
        sigma = self.get_pauli_matrix()
        sigma_t = sigma[kptpair.transitions.s1_t, kptpair.transitions.s2_t]
        f_t = kptpair.get_all(kptpair.ikpt1.f_myt)
        sigmaf_t = sigma_t * f_t

        # Calculate and add integrand
        site_quantity.array[:] += self.gs.volume * weight * np.einsum(
            't, tap -> ap', sigmaf_t, f_tap)

    @abstractmethod
    def get_pauli_matrix(self):
        """Get the desired Pauli matrix σ^μ_ss."""


class SingleParticleSiteMagnetizationCalculator(
        SingleParticleSiteSumRuleCalculator):
    r"""Calculator for the single-particle site magnetization sum rule.

    The site magnetization is calculated from the site pair density:
                 __  __
             1   \   \
    n_a^z = ‾‾‾  /   /  σ^z_ss f_nks n^a_(nks,nks)
            N_k  ‾‾  ‾‾
                 k   n,s
    """
    def create_matrix_element_calculator(self):
        return SitePairDensityCalculator(self.gs, self.context, self.sites)

    def get_pauli_matrix(self):
        return smat('z')


class SingleParticleSiteSpinSplittingCalculator(
        SingleParticleSiteMagnetizationCalculator):
    r"""Calculator for the single-particle site spin splitting sum rule.
                      __  __
                  1   \   \
    Δ^(xc)_a^z = ‾‾‾  /   /  σ^z_ss f_nks Δ^(xc,a)_(nks,nks)
                 N_k  ‾‾  ‾‾
                      k   n,s
    """
    def create_matrix_element_calculator(self):
        return SitePairSpinSplittingCalculator(
            self.gs, self.context, self.sites, rshewmin=1e-8)

    def __call__(self):
        dxc_ap = super().__call__()
        return dxc_ap * Ha  # Ha -> eV
