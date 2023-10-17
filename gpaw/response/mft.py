from __future__ import annotations

from abc import abstractmethod

import numpy as np

from gpaw.response import ResponseGroundStateAdapter, ResponseContext
from gpaw.response.frequencies import ComplexFrequencyDescriptor
from gpaw.response.chiks import ChiKSCalculator, smat
from gpaw.response.localft import LocalFTCalculator, add_LSDA_Wxc
from gpaw.response.site_kernels import SiteKernels
from gpaw.response.site_data import AtomicSites
from gpaw.response.pair_functions import SingleQPWDescriptor, PairFunction
from gpaw.response.pair_integrator import PairFunctionIntegrator
from gpaw.response.matrix_elements import (SitePairDensityCalculator,
                                           SitePairSpinSplittingCalculator)

from ase.units import Hartree


class IsotropicExchangeCalculator:
    r"""Calculator class for the Heisenberg exchange constants

    _           2
    J^ab(q) = - ‾‾ B^(xc†) K^(a†)(q) χ_KS^('+-)(q) K^b(q) B^(xc)            (1)
                V0

    calculated for an isotropic system in a plane wave representation using
    the magnetic force theorem within second order perturbation theory, see
    [J. Phys.: Condens. Matter 35 (2023) 105802].

    Entering the formula for the isotropic exchange constant at wave vector q
    between sublattice a and b is the unit cell volume V0, the functional
    derivative of the (LDA) exchange-correlation energy with respect to the
    magnitude of the magnetization B^(xc), the sublattice site kernels K^a(q)
    and K^b(q) as well as the reactive part of the static transverse magnetic
    susceptibility of the Kohn-Sham system χ_KS^('+-)(q).

    NB: To achieve numerical stability of the plane-wave implementation, we
    use instead the following expression to calculate exchange parameters:

    ˷           2
    J^ab(q) = - ‾‾ W_xc^(z†) K^(a†)(q) χ_KS^('+-)(q) K^b(q) W_xc^z          (2)
                V0

    We do this since B^(xc)(r) = -|W_xc^z(r)| is nonanalytic in points of space
    where the spin-polarization changes sign, why it is problematic to evaluate
    Eq. (1) numerically within a plane-wave representation.
    If the site partitionings only include spin-polarization of the same sign,
    Eqs. (1) and (2) should yield identical exchange parameters, but for
    antiferromagnetically aligned sites, the coupling constants differ by a
    sign.

    The site kernels encode the partitioning of real space into sites of the
    Heisenberg model. This is not a uniquely defined procedure, why the user
    has to define them externally through the SiteKernels interface."""

    def __init__(self,
                 chiks_calc: ChiKSCalculator,
                 localft_calc: LocalFTCalculator):
        """Construct the IsotropicExchangeCalculator object."""
        # Check that chiks has the assumed properties
        assumed_props = dict(
            gammacentered=True,
            nblocks=1
        )
        for key, item in assumed_props.items():
            assert getattr(chiks_calc, key) == item, \
                f'Expected chiks.{key} == {item}. '\
                f'Got: {getattr(chiks_calc, key)}'

        self.chiks_calc = chiks_calc
        self.context = chiks_calc.context

        # Check assumed properties of the LocalFTCalculator
        assert localft_calc.context is self.context
        assert localft_calc.gs is chiks_calc.gs
        self.localft_calc = localft_calc

        # W_xc^z buffer
        self._Wxc_G = None

        # χ_KS^('+-) buffer
        self._chiksr = None

    def __call__(self, q_c, site_kernels: SiteKernels, txt=None):
        """Calculate the isotropic exchange constants for a given wavevector.

        Parameters
        ----------
        q_c : nd.array
            Wave vector q in relative coordinates
        site_kernels : SiteKernels
            Site kernels instance defining the magnetic sites of the crystal
        txt : str
            Separate file to store the chiks calculation output in (optional).
            If not supplied, the output will be written to the standard text
            output location specified when initializing chiks.

        Returns
        -------
        J_abp : nd.array (dtype=complex)
            Isotropic Heisenberg exchange constants between magnetic sites a
            and b for all the site partitions p given by the site_kernels.
        """
        # Get ingredients
        Wxc_G = self.get_Wxc()
        chiksr = self.get_chiksr(q_c, txt=txt)
        qpd, chiksr_GG = chiksr.qpd, chiksr.array[0]  # array = chiksr_zGG
        V0 = qpd.gd.volume

        # Allocate an array for the exchange constants
        nsites = site_kernels.nsites
        J_pab = np.empty(site_kernels.shape + (nsites,), dtype=complex)

        # Compute exchange coupling
        for J_ab, K_aGG in zip(J_pab, site_kernels.calculate(qpd)):
            for a in range(nsites):
                for b in range(nsites):
                    J = np.conj(Wxc_G) @ np.conj(K_aGG[a]).T @ chiksr_GG \
                        @ K_aGG[b] @ Wxc_G
                    J_ab[a, b] = - 2. * J / V0

        # Transpose to have the partitions index last
        J_abp = np.transpose(J_pab, (1, 2, 0))

        return J_abp * Hartree  # Convert from Hartree to eV

    def get_Wxc(self):
        """Get B^(xc)_G from buffer."""
        if self._Wxc_G is None:  # Calculate if buffer is empty
            self._Wxc_G = self._calculate_Wxc()

        return self._Wxc_G

    def _calculate_Wxc(self):
        """Calculate the Fourier transform W_xc^z(G)."""
        # Create a plane wave descriptor encoding the plane wave basis. Input
        # q_c is arbitrary, since we are assuming that chiks.gammacentered == 1
        qpd0 = self.chiks_calc.get_pw_descriptor([0., 0., 0.])

        return self.localft_calc(qpd0, add_LSDA_Wxc)

    def get_chiksr(self, q_c, txt=None):
        """Get χ_KS^('+-)(q) from buffer."""
        q_c = np.asarray(q_c)

        # Calculate if buffer is empty or a new q-point is given
        if self._chiksr is None or not np.allclose(q_c, self._chiksr.q_c):
            self._chiksr = self._calculate_chiksr(q_c, txt=txt)

        return self._chiksr

    def _calculate_chiksr(self, q_c, txt=None):
        r"""Use the ChiKSCalculator to calculate the reactive part of the
        static Kohn-Sham susceptibility χ_KS^('+-)(q).

        First, the dynamic Kohn-Sham susceptibility

                                 __  __
                              1  \   \        f_nk↑ - f_mk+q↓
        χ_KS,GG'^+-(q,ω+iη) = ‾  /   /  ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
                              V  ‾‾  ‾‾ ħω - (ε_mk+q↓ - ε_nk↑) + iħη
                                 k  n,m
                                        x n_nk↑,mk+q↓(G+q) n_mk+q↓,nk↑(-G'-q)

        is calculated in the static limit ω=0 and without broadening η=0. Then,
        the reactive part (see [PRB 103, 245110 (2021)]) is extracted:

                              1
        χ_KS,GG'^(+-')(q,z) = ‾ [χ_KS,GG'^+-(q,z) + χ_KS,-G'-G^-+(-q,-z*)].
                              2
        """
        # Initiate new output file, if supplied
        if txt is not None:
            self.context.new_txt_and_timer(txt)

        zd = ComplexFrequencyDescriptor.from_array([0. + 0.j])
        chiks = self.chiks_calc.calculate('+-', q_c, zd)
        if np.allclose(q_c, 0.):
            chiks.symmetrize_reciprocity()

        # Take the reactive part
        chiksr = chiks.copy_reactive_part()

        return chiksr


class StaticSitePairFunction(PairFunction):
    """Data object for static site pair functions."""

    def __init__(self,
                 qpd: SingleQPWDescriptor,
                 sites: AtomicSites):
        self.qpd = qpd
        self.q_c = qpd.q_c

        self.sites = sites

        self.array = self.zeros()

    @property
    def shape(self):
        nsites = len(self.sites)
        npartitions = self.sites.npartitions
        return nsites, nsites, npartitions

    def zeros(self):
        return np.zeros(self.shape, dtype=complex)


class TwoParticleSiteSumRuleCalculator(PairFunctionIntegrator):
    r"""Calculator for two-particle site sum rules.

    For any set of site matrix elements f^a and g^b, one may define a two-
    particle site sum rule based on the lattice Fourier transformed quantity:
                     __  __   __
                 1   \   \    \   /
    ̄x_ab^z(q) = ‾‾‾  /   /    /   | σ^j_ss' (f_nks - f_n'k+qs')
                N_k  ‾‾  ‾‾   ‾‾  \                                       \
                     k  n,n' s,s'   × f^a_(nks,n'k+qs') g^b_(n'k+qs',nks) |
                                                                          /
    where σ^j is a Pauli matrix with j∊{0,+,-,z}.
    """

    def __init__(self,
                 gs: ResponseGroundStateAdapter,
                 sites: AtomicSites,
                 context: ResponseContext | None = None,
                 nblocks: int = 1,
                 nbands: int | None = None):
        """Construct the two-particle site sum rule calculator."""
        if context is None:
            context = ResponseContext()
        super().__init__(gs, context,
                         nblocks=nblocks,
                         # Disable use of symmetries for now. The sum rule site
                         # magnetization symmetries can always be derived and
                         # implemented at a later stage.
                         disable_point_group=True,
                         disable_time_reversal=True)
        self.nbands = nbands

        # Set up calculators for the f^a and g^b matrix elements
        self.sites = sites
        mecalc1, mecalc2 = self.create_matrix_element_calculators()
        self.matrix_element_calc1 = mecalc1
        self.matrix_element_calc2 = mecalc2

    @abstractmethod
    def create_matrix_element_calculators(self):
        """Create the desired site matrix element calculators."""

    def __call__(self, q_c):
        """Calculate the site sum rule for a given wave vector q_c."""
        spincomponent = self.get_spincomponent()
        transitions = self.get_band_and_spin_transitions(
            spincomponent, nbands=self.nbands, bandsummation='double')
        self.context.print(self.get_info_string(
            q_c, self.nbands, len(transitions)))

        # Set up data object (without a plane-wave representation, which is
        # irrelevant in this case)
        qpd = self.get_pw_descriptor(q_c, ecut=1e-3)
        site_pair_function = StaticSitePairFunction(qpd, self.sites)

        # Perform actual calculation
        self._integrate(site_pair_function, transitions)

        return site_pair_function.array

    @abstractmethod
    def get_spincomponent(self):
        """Define how to rotate the spins via the spin component (μν)."""

    def add_integrand(self, kptpair, weight, site_pair_function):
        r"""Add the site sum rule integrand of the outer k-point integral.

        With
                       __
                    1  \
        ̄x_ab^z(q) = ‾  /  (...)_k
                    V  ‾‾
                       k

        the integrand is given by
                     __   __
                     \    \   /
        (...)_k = V0 /    /   | σ^j_ss' (f_nks - f_n'k+qs')
                     ‾‾   ‾‾  \                                       \
                    n,n' s,s'   × f^a_(nks,n'k+qs') g^b_(n'k+qs',nks) |
                                                                      /

        where V0 is the cell volume.
        """
        # Calculate site matrix elements
        qpd = site_pair_function.qpd
        matrix_element1 = self.matrix_element_calc1(kptpair, qpd)
        if self.matrix_element_calc2 is self.matrix_element_calc1:
            matrix_element2 = matrix_element1
        else:
            matrix_element2 = self.matrix_element_calc2(kptpair, qpd)

        # Calculate the product between the Pauli matrix and the occupational
        # differences
        sigma = self.get_pauli_matrix()
        s1_myt, s2_myt = kptpair.get_local_spin_indices()
        sigma_myt = sigma[s1_myt, s2_myt]
        df_myt = kptpair.ikpt1.f_myt - kptpair.ikpt2.f_myt
        sigmadf_myt = sigma_myt * df_myt

        # Calculate integrand
        f_mytap = matrix_element1.local_array_view
        g_mytap = matrix_element2.local_array_view
        fgcc_mytabp = f_mytap[:, :, np.newaxis] * g_mytap.conj()[:, np.newaxis]
        # Sum over local transitions
        integrand_abp = np.einsum('t, tabp -> abp', sigmadf_myt, fgcc_mytabp)
        # Sum over distributed transitions
        kptpair.tblocks.blockcomm.sum(integrand_abp)

        # Add integrand to output array
        site_pair_function.array[:] += self.gs.volume * weight * integrand_abp

    @abstractmethod
    def get_pauli_matrix(self):
        """Get the desired Pauli matrix σ^j_ss'."""

    def get_info_string(self, q_c, nbands, nt):
        """Get information about the calculation"""
        info_list = ['',
                     'Calculating two-particle site sum rule with:'
                     f'    q_c: [{q_c[0]}, {q_c[1]}, {q_c[2]}]',
                     self.get_band_and_transitions_info_string(nbands, nt),
                     '',
                     self.get_basic_info_string()]
        return '\n'.join(info_list)


class TwoParticleSiteMagnetizationCalculator(TwoParticleSiteSumRuleCalculator):
    r"""Calculator for the two-particle site magnetization sum rule.

    The site magnetization can be calculated from the site pair densities via
    the following sum rule [publication in preparation]:
                     __  __
                 1   \   \
    ̄n_ab^z(q) = ‾‾‾  /   /  (f_nk↑ - f_mk+q↓) n^a_(nk↑,mk+q↓) n^b_(mk+q↓,nk↑)
                N_k  ‾‾  ‾‾
                     k   n,m

              = δ_(a,b) n_a^z

    This is directly related to the sum rule of the χ^(+-) spin component of
    the four-component susceptibility tensor.
    """
    def create_matrix_element_calculators(self):
        site_pair_density_calc = SitePairDensityCalculator(
            self.gs, self.context, self.sites)
        return site_pair_density_calc, site_pair_density_calc

    def get_spincomponent(self):
        return '+-'

    def get_pauli_matrix(self):
        return smat('+')


class TwoParticleSiteSpinSplittingCalculator(
        TwoParticleSiteMagnetizationCalculator):
    r"""Calculator for the two-particle site spin splitting sum rule.

    The site spin splitting can be calculated from the site pair density and
    site pair spin splitting via the following sum rule [publication in
    preparation]:
                          __  __
    ˍ                 1   \   \  /
    Δ^(xc)_ab^z(q) = ‾‾‾  /   /  | (f_nk↑ - f_mk+q↓)
                     N_k  ‾‾  ‾‾ \                                        \
                          k   n,m  × Δ^(xc,a)_(nk↑,mk+q↓) n^b_(mk+q↓,nk↑) |
                                                                          /
              = δ_(a,b) Δ^(xc)_a^z
    """
    def create_matrix_element_calculators(self):
        site_pair_spin_splitting_calc = SitePairSpinSplittingCalculator(
            self.gs, self.context, self.sites)
        site_pair_density_calc = SitePairDensityCalculator(
            self.gs, self.context, self.sites)
        return site_pair_spin_splitting_calc, site_pair_density_calc

    def __call__(self, *args):
        dxc_abp = super().__call__(*args)
        return dxc_abp * Hartree  # Ha -> eV
