from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from ase.units import Hartree

from gpaw.mpi import normalize_communicator
from gpaw.response.chi0 import Chi0Calculator, get_frequency_descriptor
from gpaw.response.chi0_data import Chi0Data
from gpaw.response.coulomb_kernels import CoulombKernel
from gpaw.response.density_kernels import DensityXCKernel
from gpaw.response.dyson import DysonEquation
from gpaw.response.pair import get_gs_and_context
from gpaw.response.pw_parallelization import Blocks1D

if TYPE_CHECKING:
    from gpaw.response.frequencies import FrequencyDescriptor
    from gpaw.response.groundstate import CellDescriptor


"""
On the notation in this module.

When calculating properties such as the dielectric function, EELS spectrum and
polarizability there are many inherent subtleties relating to (ir)reducible
representations and inclusion of local-field effects. For the reciprocal space
representation of the Coulomb potential, we use the following notation

v or v(q): The bare Coulomb interaction, 4π/|G+q|²

V or V(q): The specified Coulomb interaction. Will usually be either the bare
           interaction or a truncated version hereof.
ˍ    ˍ
V or V(q): The modified Coulomb interaction. Equal to V(q) for finite
           reciprocal wave vectors G > 0, but modified to exclude long-range
           interactions, that is, equal to 0 for G = 0.
"""


@dataclass
class DielectricResponse:
    """Dielectric response data for a given (q, V, xc) configuration.

    Holds the ingredients needed to compute various dielectric properties
    at a given wave vector q, for a given Coulomb interaction V(q), and
    (optionally) an exchange-correlation kernel.

    Physical quantities are accessed through methods:

        response.dielectric_function(direction)  -> ScalarResponseFunctionSet
        response.eels_spectrum(direction)         -> ScalarResponseFunctionSet
        response.polarizability(direction)        -> ScalarResponseFunctionSet
        response.dielectric_constant(direction)   -> np.ndarray
        response.dynamic_susceptibility(direction) -> ScalarResponseFunctionSet
    """
    chi0: Chi0Data
    coulomb: CoulombKernel
    xc_kernel: DensityXCKernel | None
    cd: CellDescriptor

    def __post_init__(self):
        if self.coulomb.truncation is None:
            self.bare_coulomb = self.coulomb
        else:
            self.bare_coulomb = self.coulomb.new(truncation=None)
        # When inverting the Dyson equation, we distribute frequencies globally
        blockdist = self.chi0.body.blockdist.new_distributor(nblocks='max')
        self.wblocks = Blocks1D(blockdist.blockcomm, len(self.chi0.wd))

    # ========== Public physics interface ========== #

    def dielectric_function(self, direction='x'):
        """Get the macroscopic dielectric function ε_M(q,ω).

        Calculates via the inverse dielectric function approach:

                       1
        ε (q,ω) =  ‾‾‾‾‾‾‾‾
         M         ε⁻¹(q,ω)
                    00

        along with the macroscopic dielectric function in the independent-
        particle random-phase approximation [Rev. Mod. Phys. 74, 601 (2002)],

         IPRPA
        ε (q,ω) = 1 - v(q) χ⁰(q,ω)
         M                  00

        that is, neglecting local-field and exchange-correlation effects.
        """
        vchi0_W, vchi_W = self._macroscopic_vchi_symm(direction)
        eps0_W = 1 - vchi0_W
        eps_W = 1 / (1 + vchi_W)
        return ScalarResponseFunctionSet(self.chi0.wd, eps0_W, eps_W)

    def eels_spectrum(self, direction='x'):
        """Get the macroscopic EELS spectrum.

        The spectrum is defined as

                                          1
        EELS(q,ω) ≡ -Im ε⁻¹(q,ω) = -Im ‾‾‾‾‾‾‾.
                         00            ε (q,ω)
                                        M

        In addition to the many-body spectrum, we also calculate the
        EELS spectrum in the relevant independent-particle approximation,
        here defined as

                          1
        EELS₀(ω) = -Im ‾‾‾‾‾‾‾.
                        IP
                       ε (q,ω)
                        M
        """
        _, eps0_W, eps_W = self.dielectric_function(direction).arrays
        eels0_W = -(1. / eps0_W).imag
        eels_W = -(1. / eps_W).imag
        return ScalarResponseFunctionSet(self.chi0.wd, eels0_W, eels_W)

    def polarizability(self, direction='x'):
        """Get the macroscopic polarizability α_M(q,ω).

        α_M(q,ω) = Λ/(4π) (ε_M(q,ω) - 1),

        where Λ is the nonperiodic hypervolume of the unit cell.
        When using a truncated Coulomb kernel, the bare dielectric function
        path is used instead.
        """
        if self.coulomb is not self.bare_coulomb:
            # Truncated Coulomb: use bare DF path
            if self.xc_kernel:
                raise NotImplementedError(
                    'Bare dielectric function within TDDFT has not yet been '
                    'implemented. For TDDFT, calculate the inverse DF.')
            vP_symm_wGG, vchibar_symm_wGG = self._calculate_vchi_symm(
                direction=direction, modified=True)
            vP_W = self.wblocks.all_gather(vP_symm_wGG[:, 0, 0])
            vchibar_W = self.wblocks.all_gather(vchibar_symm_wGG[:, 0, 0])
            eps0_W = 1. - vP_W
            eps_W = 1. - vchibar_W
        else:
            # No truncation: use inverse DF path
            _, eps0_W, eps_W = self.dielectric_function(direction).arrays
        L = self.cd.nonperiodic_hypervolume
        alpha0_W = L / (4 * np.pi) * (eps0_W - 1)
        alpha_W = L / (4 * np.pi) * (eps_W - 1)
        return ScalarResponseFunctionSet(self.chi0.wd, alpha0_W, alpha_W)

    def dielectric_constant(self, direction='x'):
        """Get the static dielectric constant.

        The macroscopic dielectric constant is defined as the real part of the
        dielectric function in the static limit.

        Returns
        -------
        np.ndarray
            Array of [eps0, eps] — dielectric constant without and with local
            field corrections.
        """
        return self.dielectric_function(direction).static_limit.real

    def dynamic_susceptibility(self, direction='x'):
        """Get the macroscopic components of χ(q,ω) and χ₀(q,ω)."""
        vchi0_W, vchi_W = self._macroscopic_vchi_symm(direction)
        v0 = self.bare_coulomb.V(self.chi0.qpd)[0]
        return ScalarResponseFunctionSet(
            self.chi0.wd, vchi0_W / v0, vchi_W / v0)

    # ========== Power user interface (bse.py, qeh.py) ========== #

    def get_chi0_wGG(self, direction='x'):
        """Extract χ₀(q,ω) as a plane-wave matrix.

        In the optical limit, the head and wings are projected along the
        input direction using k·p perturbation theory.
        """
        chi0 = self.chi0
        chi0_wGG = chi0.body.get_distributed_frequencies_array().copy()
        if chi0.qpd.optical_limit:
            d_v = self._normalize(direction)
            W_w = self.wblocks.myslice
            chi0_wGG[:, 0] = np.dot(d_v, chi0.chi0_WxvG[W_w, 0])
            chi0_wGG[:, :, 0] = np.dot(d_v, chi0.chi0_WxvG[W_w, 1])
            chi0_wGG[:, 0, 0] = np.dot(
                d_v, np.dot(chi0.chi0_Wvv[W_w], d_v).T)
        return chi0_wGG

    def rpa_density_response(self, direction='x', qinf_v=None):
        """Calculate the RPA susceptibility for (semi-)finite q.

        Currently this is only used by the QEH code, why we don't support a top
        level user interface.
        """
        # Extract χ₀(q,ω)
        qpd = self.chi0.qpd
        chi0_wGG = self.get_chi0_wGG(direction=direction)
        if qpd.optical_limit:
            # Restore the q-dependence of the head and wings in the q→0 limit
            assert qinf_v is not None and np.linalg.norm(qinf_v) > 0.
            d_v = self._normalize(direction)
            same_direction = np.allclose(d_v, self._normalize(qinf_v))
            if not same_direction:
                raise ValueError(
                    '`qinf_v` must be in the same direction as `direction`. '
                    f'Obtained {qinf_v=} and {direction=}')
            chi0_wGG[:, 1:, 0] *= np.dot(qinf_v, d_v)
            chi0_wGG[:, 0, 1:] *= np.dot(qinf_v, d_v)
            chi0_wGG[:, 0, 0] *= np.dot(qinf_v, d_v)**2
        # Invert Dyson equation, χ(q,ω) = [1 - χ₀(q,ω) V(q)]⁻¹ χ₀(q,ω)
        V_GG = self.coulomb.kernel(qpd, q_v=qinf_v)
        chi_wGG = self._invert_dyson_like_equation(chi0_wGG, V_GG)
        return qpd, chi_wGG, self.wblocks

    # ========== Private computation helpers ========== #

    @staticmethod
    def _normalize(direction):
        if isinstance(direction, str):
            d_v = {'x': [1, 0, 0],
                   'y': [0, 1, 0],
                   'z': [0, 0, 1]}[direction]
        else:
            d_v = direction
        d_v = np.asarray(d_v) / np.linalg.norm(d_v)
        return d_v

    def _get_coulomb_scaled_kernel(self, modified=False, Kxc_GG=None):
        """Get the Hxc kernel rescaled by the bare Coulomb potential v(q).

        Calculates
        ˷
        K(q) = v^(-1/2)(q) K_Hxc(q) v^(-1/2)(q),

        where v(q) is the bare Coulomb potential and

        K_Hxc(q) = V(q) + K_xc(q).

        When using the `modified` flag, the specified Coulomb kernel will be
        replaced with its modified analogue,
                ˍ
        V(q) -> V(q)
        """
        qpd = self.chi0.qpd
        if self.coulomb is self.bare_coulomb:
            v_G = self.coulomb.V(qpd)  # bare Coulomb interaction
            K_GG = np.eye(len(v_G), dtype=complex)
        else:
            v_G = self.bare_coulomb.V(qpd)
            V_G = self.coulomb.V(qpd)
            K_GG = np.diag(V_G / v_G)
        if modified:
            K_GG[0, 0] = 0.
        if Kxc_GG is not None:
            sqrtv_G = v_G**0.5
            K_GG += Kxc_GG / sqrtv_G / sqrtv_G[:, np.newaxis]
        return v_G, K_GG

    def _calculate_vchi_symm(self, direction='x', modified=False):
        """Calculate v^(1/2) χ v^(1/2).

        Starting from the TDDFT Dyson equation

        χ(q,ω) = χ₀(q,ω) + χ₀(q,ω) K_Hxc(q,ω) χ(q,ω),                (1)

        the Coulomb scaled susceptibility,
        ˷
        χ(q,ω) = v^(1/2)(q) χ(q,ω) v^(1/2)(q)

        can be calculated from the Dyson-like equation
        ˷        ˷         ˷       ˷      ˷
        χ(q,ω) = χ₀(q,ω) + χ₀(q,ω) K(q,ω) χ(q,ω)                     (2)

        where
        ˷
        K(q,ω) = v^(-1/2)(q) K_Hxc(q,ω) v^(-1/2)(q).

        Here v(q) refers to the bare Coulomb potential. It should be emphasized
        that invertion of (2) rather than (1) is not merely a rescaling
        excercise. In the optical q → 0 limit, the Coulomb kernel v(q) diverges
        as 1/|G+q|² while the Kohn-Sham susceptibility χ₀(q,ω) vanishes as
        |G+q|². Treating v^(1/2)(q) χ₀(q,ω) v^(1/2)(q) as a single variable,
        the effects of this cancellation can be treated accurately within k.p
        perturbation theory.
        """
        chi0_wGG = self.get_chi0_wGG(direction=direction)
        Kxc_GG = self.xc_kernel(self.chi0.qpd, chi0_wGG=chi0_wGG) \
            if self.xc_kernel else None
        v_G, K_GG = self._get_coulomb_scaled_kernel(
            modified=modified, Kxc_GG=Kxc_GG)
        # Calculate v^(1/2)(q) χ₀(q,ω) v^(1/2)(q)
        sqrtv_G = v_G**0.5
        vchi0_symm_wGG = chi0_wGG  # reuse buffer
        for w, chi0_GG in enumerate(chi0_wGG):
            vchi0_symm_wGG[w] = chi0_GG * sqrtv_G * sqrtv_G[:, np.newaxis]
        # Invert Dyson equation
        vchi_symm_wGG = self._invert_dyson_like_equation(
            vchi0_symm_wGG, K_GG, reuse_buffer=False)
        return vchi0_symm_wGG, vchi_symm_wGG

    @staticmethod
    def _invert_dyson_like_equation(in_wGG, K_GG, reuse_buffer=True):
        """Generalized Dyson equation invertion.

        Calculates

        B(q,ω) = [1 - A(q,ω) K(q)]⁻¹ A(q,ω)

        while possibly storing the output B(q,ω) in the input A(q,ω) buffer.
        """
        if reuse_buffer:
            out_wGG = in_wGG
        else:
            out_wGG = np.zeros_like(in_wGG)
        for w, in_GG in enumerate(in_wGG):
            out_wGG[w] = DysonEquation(in_GG, in_GG @ K_GG).invert()
        return out_wGG

    def _macroscopic_vchi_symm(self, direction):
        """Compute macroscopic components of v^(1/2) χ₀ v^(1/2) and
        v^(1/2) χ v^(1/2)."""
        vchi0_symm_wGG, vchi_symm_wGG = self._calculate_vchi_symm(direction)
        vchi0_W = self.wblocks.all_gather(vchi0_symm_wGG[:, 0, 0])
        vchi_W = self.wblocks.all_gather(vchi_symm_wGG[:, 0, 0])
        return vchi0_W, vchi_W

    def _polarizability_operator(self, direction='x'):
        """Calculate the polarizability operator P(q,ω).

        Depending on the theory (RPA, TDDFT, MBPT etc.), the polarizability
        operator is approximated in various ways see e.g.
        [Rev. Mod. Phys. 74, 601 (2002)].

        In RPA:
            P(q,ω) = χ₀(q,ω)

        In TDDFT:
            P(q,ω) = [1 - χ₀(q,ω) K_xc(q,ω)]⁻¹ χ₀(q,ω)
        """
        chi0_wGG = self.get_chi0_wGG(direction=direction)
        if not self.xc_kernel:  # RPA
            return chi0_wGG
        # TDDFT (in adiabatic approximations to the kernel)
        if self.chi0.qpd.optical_limit:
            raise NotImplementedError(
                'Calculation of the TDDFT dielectric function via the '
                'polarizability operator has not been implemented for the '
                'optical limit. Please calculate the inverse dielectric '
                'function instead.')
        Kxc_GG = self.xc_kernel(self.chi0.qpd, chi0_wGG=chi0_wGG)
        return self._invert_dyson_like_equation(chi0_wGG, Kxc_GG)

    def dielectric_matrix(self, direction='x'):
        """Compute the full dielectric matrix Ε(q,ω) = 1 - V(q) P(q,ω)."""
        V_GG = self.coulomb.kernel(self.chi0.qpd)
        P_wGG = self._polarizability_operator(direction=direction)
        nG = len(V_GG)
        eps_wGG = P_wGG  # reuse buffer
        for w, P_GG in enumerate(P_wGG):
            eps_wGG[w] = np.eye(nG) - V_GG @ P_GG
        return eps_wGG


# Backward compatibility alias
Chi0DysonEquations = DielectricResponse


class DielectricFunction:
    """User-facing dielectric function calculator.

    Provides convenience methods for calculating dielectric properties
    (dielectric function, EELS spectrum, polarizability, etc.) with
    automatic file output and legacy return formats.

    For the full programmatic interface, use the calculate() method
    to obtain a DielectricResponse object.
    """

    def __init__(self, calc, *,
                 frequencies=None,
                 ecut=50,
                 hilbert=True,
                 nbands=None, eta=0.2,
                 intraband=True, nblocks=1, world=None, txt=sys.stdout,
                 truncation=None,
                 qsymmetry=True,
                 integrationmode='point integration', rate=0.0,
                 eshift: float | None = None):
        """Creates a DielectricFunction object.

        calc: str
            The ground-state calculation file that the linear response
            calculation is based on.
        frequencies:
            Input parameters for frequency_grid.
            Can be an array of frequencies to evaluate the response function at
            or dictionary of parameters for build-in nonlinear grid
            (see :ref:`frequency grid`).
        ecut: float | dict
            Plane-wave cut-off or dictionary for an optional planewave
            descriptor. See response/qpd.py for details.
        hilbert: bool
            Use Hilbert transform.
        nbands: int
            Number of bands from calculation.
        eta: float
            Broadening parameter.
        intraband: bool
            Include intraband transitions.
        world: comm
            mpi communicator.
        nblocks: int
            Split matrices in nblocks blocks and distribute them G-vectors or
            frequencies over processes.
        txt: str
            Output file.
        truncation: str or None
            None for no truncation.
            '2D' for standard analytical truncation scheme.
            Non-periodic directions are determined from k-point grid
        integrationmode: str
            if == 'tetrahedron integration' then tetrahedron
            integration is performed
            if == 'point integration' then point integration is used
        eshift: float
            Shift unoccupied bands
        """
        world = normalize_communicator(world)
        gs, context = get_gs_and_context(calc, txt, world, timer=None)
        wd = get_frequency_descriptor(frequencies, gs=gs, nbands=nbands)

        if integrationmode is None:
            raise DeprecationWarning(
                "Please use `integrationmode='point integration'` instead")
            integrationmode = 'point integration'

        self.chi0calc = Chi0Calculator(
            gs, context, nblocks=nblocks,
            wd=wd,
            ecut=ecut, nbands=nbands, eta=eta,
            hilbert=hilbert,
            intraband=intraband,
            qsymmetry=qsymmetry,
            integrationmode=integrationmode,
            rate=rate, eshift=eshift
        )
        self.gs = gs
        self.context = context
        self.truncation = truncation
        self._chi0cache: dict[tuple[str, ...], Chi0Data] = {}

    def _get_chi0(self, q_c: list | np.ndarray) -> Chi0Data:
        """Get the Kohn-Sham susceptibility χ₀(q,ω) for input wave vector q.

        Keeps a cache of χ₀ for the latest calculated wave vector, thus
        allowing for investigation of multiple dielectric properties,
        Coulomb truncations, xc kernels etc. without recalculating χ₀.
        """
        q_key = [f'{q:.10f}' for q in q_c]
        key = tuple(q_key)
        if key not in self._chi0cache:
            self._chi0cache.clear()
            self._chi0cache[key] = self.chi0calc.calculate(q_c)
            self.context.write_timer()
        return self._chi0cache[key]

    def calculate(self,
                  q_c: list | np.ndarray = [0, 0, 0],
                  truncation: str | None = None,
                  xc: str = 'RPA',
                  **xckwargs
                  ) -> DielectricResponse:
        """Calculate the dielectric response at wave vector q.

        Returns a DielectricResponse object from which physical quantities
        (dielectric function, EELS, polarizability, etc.) can be extracted.

        Parameters
        ----------
        q_c : list or np.ndarray
            Wave vector in scaled coordinates.
        truncation : str or None
            Truncation of the Coulomb kernel.
        xc : str
            Exchange-correlation kernel for LR-TDDFT calculations.
            If xc == 'RPA', the dielectric response is treated in the random
            phase approximation.
        **xckwargs
            Additional parameters for the chosen xc kernel.
        """
        chi0 = self._get_chi0(q_c)
        coulomb = CoulombKernel.from_gs(self.gs, truncation=truncation)
        if xc == 'RPA':
            xc_kernel = None
        else:
            xc_kernel = DensityXCKernel.from_functional(
                self.gs, self.context, functional=xc, **xckwargs)
        return DielectricResponse(chi0, coulomb, xc_kernel, self.gs.cd)

    # Backward compatibility alias
    get_chi0_dyson_eqs = calculate

    def get_frequencies(self) -> np.ndarray:
        """Return frequencies (in eV) that the χ is evaluated on."""
        return self.chi0calc.wd.omega_w * Hartree

    def get_dielectric_function(self,
                                q_c: list | np.ndarray = [0, 0, 0],
                                direction='x',
                                xc='RPA',
                                filename='df.csv',
                                **xckwargs):
        """Calculate the dielectric function.

        Returns (df_NLFC_w, df_LFC_w) and writes to filename.
        """
        response = self.calculate(
            q_c, truncation=self.truncation, xc=xc, **xckwargs)
        df = response.dielectric_function(direction=direction)
        if filename:
            df.write(filename, comm=self.context.comm)
        return df.unpack()

    def get_eels_spectrum(self,
                          q_c: list | np.ndarray = [0, 0, 0],
                          direction='x',
                          xc='RPA',
                          filename='eels.csv',
                          **xckwargs):
        """Calculate the macroscopic EELS spectrum.

        Returns (eels0_w, eels_w) and writes to filename.
        """
        response = self.calculate(
            q_c, truncation=self.truncation, xc=xc, **xckwargs)
        eels = response.eels_spectrum(direction=direction)
        if filename:
            eels.write(filename, comm=self.context.comm)
        return eels.unpack()

    def get_dynamic_susceptibility(self,
                                   q_c: list | np.ndarray = [0, 0, 0],
                                   direction='x',
                                   xc='ALDA',
                                   filename='chiM_w.csv',
                                   **xckwargs):
        """Calculate the dynamic susceptibility."""
        response = self.calculate(
            q_c, truncation=self.truncation, xc=xc, **xckwargs)
        dynsus = response.dynamic_susceptibility(direction=direction)
        if filename:
            dynsus.write(filename, comm=self.context.comm)
        return dynsus.unpack()

    def get_polarizability(self, q_c: list | np.ndarray = [0, 0, 0],
                           direction='x', filename='polarizability.csv',
                           **xckwargs):
        """Calculate the macroscopic polarizability.

        Returns (alpha0_w, alpha_w) and writes to filename.
        """
        response = self.calculate(
            q_c, truncation=self.truncation, **xckwargs)
        pol = response.polarizability(direction=direction)
        if filename:
            pol.write(filename, comm=self.context.comm)
        return pol.unpack()

    def get_macroscopic_dielectric_constant(self, xc='RPA', direction='x'):
        """Calculate the macroscopic dielectric constant.

        The macroscopic dielectric constant is defined as the real part of the
        dielectric function in the static limit.

        Returns:
        --------
        eps0: float
            Dielectric constant without local field corrections.
        eps: float
            Dielectric constant with local field correction. (RPA, ALDA)
        """
        response = self.calculate(xc=xc)
        return response.dielectric_constant(direction=direction)


# ----- Serialized dataclasses and IO ----- #


@dataclass
class ScalarResponseFunctionSet:
    """A set of scalar response functions rf₀(ω) and rf(ω)."""
    wd: FrequencyDescriptor
    rf0_w: np.ndarray
    rf_w: np.ndarray

    @property
    def arrays(self):
        return self.wd.omega_w * Hartree, self.rf0_w, self.rf_w

    def unpack(self):
        # Legacy feature to support old DielectricFunction output format
        # ... to be deprecated ...
        return self.rf0_w, self.rf_w

    def write(self, filename, *, comm=None):
        comm = normalize_communicator(comm)
        if comm.rank == 0:
            write_response_function(filename, *self.arrays)

    @property
    def static_limit(self):
        """Return the value of the response functions in the static limit."""
        w0 = np.argmin(np.abs(self.wd.omega_w))
        assert abs(self.wd.omega_w[w0]) < 1e-8
        return np.array([self.rf0_w[w0], self.rf_w[w0]])


def write_response_function(filename, omega_w, rf0_w, rf_w):
    with open(filename, 'w') as fd:
        for omega, rf0, rf in zip(omega_w, rf0_w, rf_w):
            if rf0_w.dtype == complex:
                print('%.6f, %.6f, %.6f, %.6f, %.6f' %
                      (omega, rf0.real, rf0.imag, rf.real, rf.imag),
                      file=fd)
            else:
                print(f'{omega:.6f}, {rf0:.6f}, {rf:.6f}', file=fd)


def read_response_function(filename):
    """Read a stored response function file"""
    d = np.loadtxt(filename, delimiter=',')
    omega_w = np.array(d[:, 0], float)

    if d.shape[1] == 3:
        # Real response function
        rf0_w = np.array(d[:, 1], float)
        rf_w = np.array(d[:, 2], float)
    elif d.shape[1] == 5:
        rf0_w = np.array(d[:, 1], complex)
        rf0_w.imag = d[:, 2]
        rf_w = np.array(d[:, 3], complex)
        rf_w.imag = d[:, 4]
    else:
        raise ValueError(f'Unexpected array dimension {d.shape}')

    return omega_w, rf0_w, rf_w
