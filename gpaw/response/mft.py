from __future__ import annotations

# General modules
from abc import abstractmethod

import numpy as np

# GPAW modules
from gpaw.sphere.integrate import (integrate_lebedev,
                                   radial_truncation_function,
                                   spherical_truncation_function_collection,
                                   default_spherical_drcut,
                                   find_volume_conserving_lambd)

from gpaw.response import ResponseGroundStateAdapter, ResponseContext
from gpaw.response.frequencies import ComplexFrequencyDescriptor
from gpaw.response.chiks import ChiKSCalculator, smat
from gpaw.response.localft import (LocalFTCalculator,
                                   add_LSDA_Wxc,
                                   add_spin_polarization,
                                   add_LSDA_spin_splitting)
from gpaw.response.site_kernels import SiteKernels
from gpaw.response.pair_functions import SingleQPWDescriptor, PairFunction
from gpaw.response.pair_integrator import PairFunctionIntegrator
from gpaw.response.matrix_elements import (SiteMatrixElementCalculator,
                                           SitePairDensityCalculator,
                                           SitePairSpinSplittingCalculator)

# ASE modules
from ase.units import Hartree, Bohr

from ase.neighborlist import natural_cutoffs, build_neighbor_list


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


class AtomicSites:
    """Object defining a set of spherical atomic sites."""

    def __init__(self, indices, radii):
        """Construct the AtomicSites.

        Parameters
        ----------
        indices : 1D array-like
            Atomic index A for each site index a.
        radii : 2D array-like
            Atomic radius rc for each site index a and partitioning p.
        """
        self.A_a = np.asarray(indices)
        assert self.A_a.ndim == 1
        assert len(np.unique(self.A_a)) == len(self.A_a)

        # Parse the input atomic radii
        rc_ap = np.asarray(radii)
        assert rc_ap.ndim == 2
        assert rc_ap.shape[0] == len(self.A_a)
        # Convert radii to internal units (Å to Bohr)
        self.rc_ap = rc_ap / Bohr

        self.npartitions = self.rc_ap.shape[1]
        self.shape = rc_ap.shape

    def __len__(self):
        return len(self.A_a)


class AtomicSiteData:
    r"""Data object for a set of spherical atomic sites."""

    def __init__(self, gs: ResponseGroundStateAdapter, sites: AtomicSites):
        """Extract atomic site data from a ground state adapter."""
        assert self._in_valid_site_radii_range(gs, sites), \
            'Please provide site radii in the valid range, see '\
            'AtomicSiteData.valid_site_radii_range()'
        self.sites = sites

        # Extract the scaled positions and micro_setups for each atomic site
        self.spos_ac = gs.spos_ac[sites.A_a]
        self.micro_setup_a = [gs.micro_setups[A] for A in sites.A_a]

        # Extract pseudo density on the fine real-space grid
        self.finegd = gs.finegd
        self.nt_sr = gs.nt_sr

        # Set up the atomic truncation functions which define the sites based
        # on the coarse real-space grid
        self.gd = gs.gd
        self.drcut = default_spherical_drcut(self.gd)
        self.lambd_ap = np.array(
            [[find_volume_conserving_lambd(rcut, self.drcut)
              for rcut in rc_p] for rc_p in sites.rc_ap])
        self.stfc = spherical_truncation_function_collection(
            self.finegd, self.spos_ac, sites.rc_ap, self.drcut, self.lambd_ap)

    @staticmethod
    def _valid_site_radii_range(gs):
        """For each atom in gs, determine the valid site radii range in Bohr.

        The lower bound is determined by the spherical truncation width, when
        truncating integrals on the real-space grid.
        The upper bound is determined by the distance to the nearest
        augmentation sphere.
        """
        atoms = gs.atoms
        drcut = default_spherical_drcut(gs.gd)
        rmin_A = np.array([drcut / 2] * len(atoms))

        # Find neighbours based on covalent radii
        cutoffs = natural_cutoffs(atoms, mult=2)
        neighbourlist = build_neighbor_list(atoms, cutoffs,
                                            self_interaction=False)
        # Determine rmax for each atom
        augr_A = gs.get_aug_radii()
        rmax_A = []
        for A in range(len(atoms)):
            pos = atoms.positions[A]
            # Calculate the distance to the augmentation sphere of each
            # neighbour
            aug_distances = []
            for An, offset in zip(*neighbourlist.get_neighbors(A)):
                posn = atoms.positions[An] + offset @ atoms.get_cell()
                dist = np.linalg.norm(posn - pos) / Bohr  # Å -> Bohr
                aug_dist = dist - augr_A[An]
                assert aug_dist > 0.
                aug_distances.append(aug_dist)
            # In order for PAW corrections to be valid, we need a sphere of
            # radius rcut not to overlap with any neighbouring augmentation
            # spheres
            rmax_A.append(min(aug_distances))
        rmax_A = np.array(rmax_A)

        return rmin_A, rmax_A

    @staticmethod
    def valid_site_radii_range(gs):
        """Get the valid site radii for all atoms in a given ground state."""
        rmin_A, rmax_A = AtomicSiteData._valid_site_radii_range(gs)
        # Convert to external units (Bohr to Å)
        return rmin_A * Bohr, rmax_A * Bohr

    @staticmethod
    def _in_valid_site_radii_range(gs, sites):
        rmin_A, rmax_A = AtomicSiteData._valid_site_radii_range(gs)
        for a, A in enumerate(sites.A_a):
            if not np.all(
                    np.logical_and(
                        sites.rc_ap[a] > rmin_A[A] - 1e-8,
                        sites.rc_ap[a] < rmax_A[A] + 1e-8)):
                return False
        return True

    def calculate_magnetic_moments(self):
        """Calculate the magnetic moments at each atomic site."""
        magmom_ap = self.integrate_local_function(add_spin_polarization)
        return magmom_ap

    def calculate_spin_splitting(self):
        r"""Calculate the spin splitting Δ^(xc) for each atomic site."""
        dxc_ap = self.integrate_local_function(add_LSDA_spin_splitting)
        return dxc_ap * Hartree  # return the splitting in eV

    def integrate_local_function(self, add_f):
        r"""Integrate a local function f[n](r) = f(n(r)) over the atomic sites.

        For every site index a and partitioning p, the integral is defined via
        a smooth truncation function θ(|r-r_a|<rc_ap):

               /
        f_ap = | dr θ(|r-r_a|<rc_ap) f(n(r))
               /
        """
        out_ap = np.zeros(self.sites.shape, dtype=float)
        self._integrate_pseudo_contribution(add_f, out_ap)
        self._integrate_paw_correction(add_f, out_ap)
        return out_ap

    def _integrate_pseudo_contribution(self, add_f, out_ap):
        """Calculate the pseudo contribution to the atomic site integrals.

        For local functions of the density, the pseudo contribution is
        evaluated by a numerical integration on the real-space grid:

        ̰       /
        f_ap = | dr θ(|r-r_a|<rc_ap) f(ñ(r))
               /
        """
        # Evaluate the local function on the real-space grid
        ft_r = self.finegd.zeros()
        add_f(self.finegd, self.nt_sr, ft_r)

        # Integrate θ(|r-r_a|<rc_ap) f(ñ(r))
        ftdict_ap = {a: np.empty(self.sites.npartitions)
                     for a in range(len(self.sites))}
        self.stfc.integrate(ft_r, ftdict_ap)

        # Add pseudo contribution to the output array
        for a in range(len(self.sites)):
            out_ap[a] += ftdict_ap[a]

    def _integrate_paw_correction(self, add_f, out_ap):
        """Calculate the PAW correction to an atomic site integral.

        The PAW correction is evaluated on the atom centered radial grid, using
        the all-electron and pseudo densities generated from the partial waves:

                /
        Δf_ap = | r^2 dr θ(r<rc_ap) [f(n_a(r)) - f(ñ_a(r))]
                /
        """
        for a, (micro_setup, rc_p, lambd_p) in enumerate(zip(
                self.micro_setup_a, self.sites.rc_ap, self.lambd_ap)):
            # Evaluate the PAW correction and integrate angular components
            df_ng = micro_setup.evaluate_paw_correction(add_f)
            df_g = integrate_lebedev(df_ng)
            for p, (rcut, lambd) in enumerate(zip(rc_p, lambd_p)):
                # Evaluate the smooth truncation function
                theta_g = radial_truncation_function(
                    micro_setup.rgd.r_g, rcut, self.drcut, lambd)
                # Integrate θ(r) Δf(r) on the radial grid
                out_ap[a, p] += micro_setup.rgd.integrate_trapz(df_g * theta_g)


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
        self.matrix_element_calc1: SiteMatrixElementCalculator | None = None
        self.matrix_element_calc2: SiteMatrixElementCalculator | None = None

    def __call__(self, q_c, atomic_site_data: AtomicSiteData):
        """Calculate the site sum rule for a given wave vector q_c."""
        # Set up calculators for the f^a and g^b matrix elements
        mecalc1, mecalc2 = self.create_matrix_element_calculators(
            atomic_site_data)
        self.matrix_element_calc1 = mecalc1
        self.matrix_element_calc2 = mecalc2

        spincomponent = self.get_spincomponent()
        transitions = self.get_band_and_spin_transitions(
            spincomponent, nbands=self.nbands, bandsummation='double')
        self.context.print(self.get_info_string(
            q_c, self.nbands, len(transitions)))

        # Set up data object (without a plane-wave representation, which is
        # irrelevant in this case)
        qpd = self.get_pw_descriptor(q_c, ecut=1e-3)
        site_pair_function = StaticSitePairFunction(
            qpd, atomic_site_data.sites)

        # Perform actual calculation
        self._integrate(site_pair_function, transitions)

        return site_pair_function.array

    @abstractmethod
    def create_matrix_element_calculators(self, atomic_site_data):
        """Create the desired site matrix element calculators."""

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
    def create_matrix_element_calculators(self, atomic_site_data):
        site_pair_density_calc = SitePairDensityCalculator(
            self.gs, self.context, atomic_site_data)
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
    def create_matrix_element_calculators(self, atomic_site_data):
        site_pair_spin_splitting_calc = SitePairSpinSplittingCalculator(
            self.gs, self.context, atomic_site_data)
        site_pair_density_calc = SitePairDensityCalculator(
            self.gs, self.context, atomic_site_data)
        return site_pair_spin_splitting_calc, site_pair_density_calc

    def __call__(self, *args):
        dxc_abp = super().__call__(*args)
        return dxc_abp * Hartree  # Ha -> eV
