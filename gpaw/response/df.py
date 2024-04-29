from __future__ import annotations
from dataclasses import dataclass
from math import pi
import sys

import numpy as np
from ase.units import Hartree, Bohr

import gpaw.mpi as mpi

from gpaw.response.context import ResponseContext
from gpaw.response.coulomb_kernels import CoulombKernel
from gpaw.response.density_kernels import get_density_xc_kernel
from gpaw.response.chi0 import Chi0Calculator, get_frequency_descriptor
from gpaw.response.chi0_data import Chi0Data
from gpaw.response.pair import get_gs_and_context
from gpaw.response.pair_functions import SingleQPWDescriptor

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpaw.response.frequencies import FrequencyDescriptor


@dataclass
class Chi0DysonEquation:
    chi0: Chi0Data
    df: 'DielectricFunctionCalculator'

    def __post_init__(self):
        self.gs = self.df.gs
        self.context = self.df.context
        self.coulomb = self.df.coulomb
        self.blocks1d = self.df.blocks1d

    def chi(self, xc='RPA', direction='x', return_VchiV=True, q_v=None,
            rshelmax=-1, rshewmin=None):
        """Returns qpd, chi0 and chi0, possibly in v^1/2 chi v^1/2 format.

        The truncated Coulomb interaction is included as
        v^-1/2 v_t v^-1/2. This is in order to conform with
        the head and wings of chi0, which is treated specially for q=0.

        Parameters
        ----------
        rshelmax : int or None
            Expand kernel in real spherical harmonics inside augmentation
            spheres. If None, the kernel will be calculated without
            augmentation. The value of rshelmax indicates the maximum index l
            to perform the expansion in (l < 6).
        rshewmin : float or None
            If None, the kernel correction will be fully expanded up to the
            chosen lmax. Given as a float, (0 < rshewmin < 1) indicates what
            coefficients to use in the expansion. If any coefficient
            contributes with less than a fraction of rshewmin on average,
            it will not be included.
        """
        chi0 = self.chi0
        qpd = chi0.qpd
        chi0_wGG = chi0.body.get_distributed_frequencies_array().copy()

        coulomb_bare = CoulombKernel.from_gs(self.gs, truncation=None)
        Kbare_G = coulomb_bare.V(qpd=qpd, q_v=q_v)  # np.ndarray
        sqrtV_G = Kbare_G**0.5

        nG = len(sqrtV_G)

        Ktrunc_G = self.coulomb.V(qpd=qpd, q_v=q_v)

        if self.coulomb.truncation is None:
            K_GG = np.eye(nG, dtype=complex)
        else:
            K_GG = np.diag(Ktrunc_G / Kbare_G)

        # kd: KPointDescriptor object from gpaw.kpt_descriptor
        if qpd.kd.gamma:
            if isinstance(direction, str):
                d_v = {'x': [1, 0, 0],
                       'y': [0, 1, 0],
                       'z': [0, 0, 1]}[direction]
            else:
                d_v = direction
            d_v = np.asarray(d_v) / np.linalg.norm(d_v)
            W = self.blocks1d.myslice  # slice object for this process.
            #  used to distribute the calculation when run in parallel.
            chi0_wGG[:, 0] = np.dot(d_v, chi0.chi0_WxvG[W, 0])
            chi0_wGG[:, :, 0] = np.dot(d_v, chi0.chi0_WxvG[W, 1])
            chi0_wGG[:, 0, 0] = np.dot(d_v, np.dot(chi0.chi0_Wvv[W], d_v).T)

        if xc != 'RPA':
            Kxc_GG = get_density_xc_kernel(qpd,
                                           self.gs, self.context,
                                           functional=xc,
                                           chi0_wGG=chi0_wGG)
            K_GG += Kxc_GG / sqrtV_G / sqrtV_G[:, np.newaxis]

        # Invert Dyson eq.
        chi_wGG = []
        for chi0_GG in chi0_wGG:
            """v^1/2 chi0 V^1/2"""
            chi0_GG[:] = chi0_GG * sqrtV_G * sqrtV_G[:, np.newaxis]
            chi_GG = np.dot(np.linalg.inv(np.eye(nG) -
                                          np.dot(chi0_GG, K_GG)),
                            chi0_GG)
            if not return_VchiV:
                chi0_GG /= sqrtV_G * sqrtV_G[:, np.newaxis]
                chi_GG /= sqrtV_G * sqrtV_G[:, np.newaxis]
            chi_wGG.append(chi_GG)

        if len(chi_wGG):
            chi_wGG = np.array(chi_wGG)
        else:
            chi_wGG = np.zeros((0, nG, nG), complex)

        return ChiData(self, qpd, chi0_wGG, np.array(chi_wGG))

    def dielectric_matrix(self, xc='RPA', direction='x', symmetric=True,
                          calculate_chi=False, q_v=None):
        r"""Returns the symmetrized dielectric matrix.

        ::

            \tilde\epsilon_GG' = v^{-1/2}_G \epsilon_GG' v^{1/2}_G',

        where::

            epsilon_GG' = 1 - v_G * P_GG' and P_GG'

        is the polarization.

        ::

            In RPA:   P = chi^0
            In TDDFT: P = (1 - chi^0 * f_xc)^{-1} chi^0

        in addition to RPA one can use the kernels, ALDA, Bootstrap and
        LRalpha (long-range kerne), where alpha is a user specified parameter
        (for example xc='LR0.25')

        The head of the inverse symmetrized dielectric matrix is equal
        to the head of the inverse dielectric matrix (inverse dielectric
        function)"""

        chi0 = self.chi0
        qpd = chi0.qpd
        chi0_wGG = chi0.body.get_distributed_frequencies_array().copy()

        K_G = self.coulomb.sqrtV(qpd=qpd, q_v=q_v)
        nG = len(K_G)

        if qpd.kd.gamma:
            if isinstance(direction, str):
                d_v = {'x': [1, 0, 0],
                       'y': [0, 1, 0],
                       'z': [0, 0, 1]}[direction]
            else:
                d_v = direction

            d_v = np.asarray(d_v) / np.linalg.norm(d_v)
            W = self.blocks1d.myslice
            chi0_wGG[:, 0] = np.dot(d_v, chi0.chi0_WxvG[W, 0])
            chi0_wGG[:, :, 0] = np.dot(d_v, chi0.chi0_WxvG[W, 1])
            chi0_wGG[:, 0, 0] = np.dot(d_v, np.dot(chi0.chi0_Wvv[W], d_v).T)
            if q_v is not None:
                print('Restoring q dependence of head and wings of chi0')
                chi0_wGG[:, 1:, 0] *= np.dot(q_v, d_v)
                chi0_wGG[:, 0, 1:] *= np.dot(q_v, d_v)
                chi0_wGG[:, 0, 0] *= np.dot(q_v, d_v)**2

        if xc != 'RPA':
            Kxc_GG = get_density_xc_kernel(qpd,
                                           self.gs, self.context,
                                           functional=xc,
                                           chi0_wGG=chi0_wGG)

        if calculate_chi:
            chi_wGG = []

        for chi0_GG in chi0_wGG:
            if xc == 'RPA':
                P_GG = chi0_GG
            else:
                P_GG = np.dot(np.linalg.inv(np.eye(nG) -
                                            np.dot(chi0_GG, Kxc_GG)),
                              chi0_GG)
            if symmetric:
                e_GG = np.eye(nG) - P_GG * K_G * K_G[:, np.newaxis]
            else:
                K_GG = (K_G**2 * np.ones([nG, nG])).T
                e_GG = np.eye(nG) - P_GG * K_GG

            if calculate_chi:
                K_GG = np.diag(K_G**2)
                if xc != 'RPA':
                    K_GG += Kxc_GG
                chi_wGG.append(np.dot(np.linalg.inv(np.eye(nG) -
                                                    np.dot(chi0_GG, K_GG)),
                                      chi0_GG))
            chi0_GG[:] = e_GG

        # chi0_wGG is now the dielectric matrix
        if calculate_chi:
            if len(chi_wGG):
                chi_wGG = np.array(chi_wGG)
            else:
                chi_wGG = np.zeros((0, nG, nG), complex)

        if not calculate_chi:
            return DielectricMatrixData(self, chi0_wGG=chi0_wGG)
        else:
            # chi_wGG is the full density response function..
            return DielectricMatrixData(self, qpd=qpd, chi0_wGG=chi0_wGG,
                                        chi_wGG=chi_wGG)


@dataclass
class ChiData:
    dyson: Chi0DysonEquation
    qpd: object
    chi0_wGG: np.ndarray
    chi_wGG: np.ndarray

    def unpack(self):
        return (self.qpd, self.chi0_wGG, self.chi_wGG)

    def dynamic_susceptibility(self):
        """Calculate the dynamic susceptibility.

        Returns macroscopic(could be generalized?) dynamic susceptibility:
        chiM0_w, chiM_w = DielectricFunction.get_dynamic_susceptibility()
        """
        rf0_w = np.zeros(len(self.chi_wGG), dtype=complex)
        rf_w = np.zeros(len(self.chi_wGG), dtype=complex)

        for w, (chi0_GG, chi_GG) in enumerate(zip(self.chi0_wGG,
                                                  self.chi_wGG)):
            rf0_w[w] = chi0_GG[0, 0]
            rf_w[w] = chi_GG[0, 0]

        rf0_w = self.dyson.df.collect(rf0_w)
        rf_w = self.dyson.df.collect(rf_w)

        return DynamicSusceptibility(self.wd, rf0_w, rf_w)

    @property
    def wd(self):
        return self.dyson.df.wd

    def eels_spectrum(self):
        r"""Calculate EELS spectrum. By default, generate a file 'eels.csv'.

        EELS spectrum is obtained from the imaginary part of the
        density response function as, EELS(\omega) = - 4 * \pi / q^2 Im \chi.
        Returns EELS spectrum without and with local field corrections:

        df_NLFC_w, df_LFC_w = DielectricFunction.get_eels_spectrum()"""

        # Calculate V^1/2 \chi V^1/2
        Vchi0_wGG = self.chi0_wGG  # askhl: so what's with the V^1/2?
        Vchi_wGG = self.chi_wGG

        # Calculate eels = -Im 4 \pi / q^2  \chi
        eels_NLFC_w = -(1. / (1. - Vchi0_wGG[:, 0, 0])).imag
        eels_LFC_w = -Vchi_wGG[:, 0, 0].imag

        eels_NLFC_w = self.dyson.df.collect(eels_NLFC_w)
        eels_LFC_w = self.dyson.df.collect(eels_LFC_w)
        return EELSSpectrum(self.dyson.context, self.wd,
                            eels_NLFC_w, eels_LFC_w)


@dataclass
class DynamicSusceptibility:
    wd: FrequencyDescriptor
    rf0_w: np.ndarray
    rf_w: np.ndarray

    def unpack(self):
        return self.rf0_w, self.rf_w

    def write(self, filename):
        if filename is not None and mpi.rank == 0:
            write_response_function(
                filename, self.wd.omega_w * Hartree, self.rf0_w, self.rf_w)


@dataclass
class EELSSpectrum:
    context: ResponseContext
    wd: FrequencyDescriptor
    eels_NLFC_w: np.ndarray
    eels_LFC_w: np.ndarray

    def unpack(self):
        return self.eels_NLFC_w, self.eels_LFC_w

    def write(self, filename):
        if filename is not None and self.context.comm.rank == 0:
            write_response_function(filename, self.wd.omega_w * Hartree,
                                    self.eels_NLFC_w, self.eels_LFC_w)


@dataclass
class DielectricMatrixData:
    dyson: Chi0DysonEquation
    qpd: SingleQPWDescriptor | None = None
    chi0_wGG: np.ndarray | None = None
    chi_wGG: np.ndarray | None = None

    def unpack(self):
        # (This has the (inconsistent) return types of the old API.)
        if self.qpd is None:
            return self.chi0_wGG
        return (self.qpd, self.chi0_wGG, self.chi_wGG)

    def dielectric_function(self):
        """Calculate the dielectric function.

        Returns dielectric function without and with local field correction:
        df_NLFC_w, df_LFC_w = DielectricFunction.get_dielectric_function()
        """
        e_wGG = self.chi0_wGG  # XXX what's with the names here?
        df_NLFC_w = np.zeros(len(e_wGG), dtype=complex)
        df_LFC_w = np.zeros(len(e_wGG), dtype=complex)

        for w, e_GG in enumerate(e_wGG):
            df_NLFC_w[w] = e_GG[0, 0]
            df_LFC_w[w] = 1 / np.linalg.inv(e_GG)[0, 0]

        df_NLFC_w = self.dyson.df.collect(df_NLFC_w)
        df_LFC_w = self.dyson.df.collect(df_LFC_w)

        return DielectricFunctionData(self.dyson.df.wd, df_NLFC_w, df_LFC_w)


@dataclass
class Polarizability:
    context: ResponseContext
    wd: FrequencyDescriptor
    alpha0_w: np.ndarray
    alpha_w: np.ndarray

    def unpack(self):
        return self.alpha0_w, self.alpha_w

    def write(self, filename):
        if filename is not None and self.context.comm.rank == 0:
            write_response_function(filename, self.wd.omega_w * Hartree,
                                    self.alpha0_w, self.alpha_w)


@dataclass
class DielectricFunctionData:
    wd: FrequencyDescriptor
    df_NLFC_w: np.ndarray
    df_LFC_w: np.ndarray

    def unpack(self):
        return self.df_NLFC_w, self.df_LFC_w

    def write(self, filename):
        if filename is not None and mpi.rank == 0:
            write_response_function(filename, self.wd.omega_w * Hartree,
                                    self.df_NLFC_w, self.df_LFC_w)

    @property
    def eps0(self):
        return self.df_NLFC_w[0].real

    @property
    def eps(self):
        return self.df_LFC_w[0].real


class DielectricFunctionCalculator:
    def __init__(self, wd: FrequencyDescriptor,
                 chi0calc: Chi0Calculator, truncation: str | None):
        from gpaw.response.pw_parallelization import Blocks1D
        self.wd = wd

        self.chi0calc = chi0calc

        self.coulomb = CoulombKernel.from_gs(self.gs, truncation=truncation)

        # context: ResponseContext object from gpaw.response.context
        self.context = chi0calc.context

        # context.comm : _Communicator object from gpaw.mpi
        self.blocks1d = Blocks1D(self.context.comm, len(self.wd))

        self._chi0cache: dict = {}

    @property
    def gs(self):
        # gs: ResponseGroundStateAdapter from gpaw.response.groundstate
        return self.chi0calc.gs

    def calculate_chi0(self, q_c: list | np.ndarray):
        """Calculates the response function.

        Calculate the response function for a specific momentum.

        q_c: [float, float, float]
            The momentum wavevector.
        """

        # We cache the computed data since chi0 may otherwise be redundantly
        # calculated e.g. if the user calculates multiple directions.
        #
        # May be called multiple times with same q_c, and we want to
        # be able to recognize previous seen values of q_c.
        # We do this by rounding and converting to string with fixed
        # precision (so not very elegant).
        q_key = [f'{q:.10f}' for q in q_c]
        key = tuple(q_key)

        if key not in self._chi0cache:
            # We assume that the caller will trigger this multiple
            # times with the same qpoint, then several times with
            # another qpoint, etc.  If that's true, then we
            # need to cache no more than one qpoint at a time.
            # Thus to save memory, we clear the cache here.
            #
            # This should be replaced with something more reliable,
            # such as having the caller manage things more explicitly.
            #
            # See https://gitlab.com/gpaw/gpaw/-/issues/662
            #
            # In conclusion, delete the cache now:
            self._chi0cache.clear()

            # cache Chi0Data from gpaw.response.chi0_data
            self._chi0cache[key] = Chi0DysonEquation(
                self.chi0calc.calculate(q_c), self)
            self.context.write_timer()

        return self._chi0cache[key]

    def collect(self, a_w: np.ndarray) -> np.ndarray:
        # combines array from sub-processes into one.
        return self.blocks1d.all_gather(a_w)

    def get_frequencies(self) -> np.ndarray:
        """ Return frequencies that Chi is evaluated on"""
        return self.wd.omega_w * Hartree

    def _new_chi(self, xc='RPA', q_c=[0, 0, 0], **kwargs):
        return self.calculate_chi0(q_c).chi(xc=xc, **kwargs)

    def get_chi(self, *args, **kwargs):
        return self._new_chi(*args, **kwargs).unpack()

    def _new_dynamic_susceptibility(self, xc='ALDA', q_c=[0, 0, 0], *,
                                    filename='chiM_w.csv', **kwargs):
        chi0 = self.calculate_chi0(q_c)
        chi = chi0.chi(xc=xc, return_VchiV=False, **kwargs)
        dynsus = chi.dynamic_susceptibility()
        dynsus.write(filename)
        return dynsus

    def _new_dielectric_function(self, *args, filename='df.csv', **kwargs):
        dm = self._new_dielectric_matrix(*args, **kwargs)
        df = dm.dielectric_function()
        df.write(filename)
        return df

    def _new_dielectric_matrix(self, xc='RPA', q_c=[0, 0, 0], **kwargs):
        chi0 = self.calculate_chi0(q_c)
        return chi0.dielectric_matrix(xc=xc, **kwargs)

    def get_dynamic_susceptibility(self, *args, **kwargs):
        return self._new_dynamic_susceptibility(*args, **kwargs).unpack()

    def get_dielectric_matrix(self, *args, **kwargs):
        return self._new_dielectric_matrix(*args, **kwargs).unpack()

    def get_dielectric_function(self, *args, **kwargs):
        """..."""
        return self._new_dielectric_function(*args, **kwargs).unpack()

    def get_macroscopic_dielectric_constant(self, xc='RPA',
                                            direction='x', q_v=None):
        """Calculate macroscopic dielectric constant.

        Returns eM_NLFC and eM_LFC.

        Macroscopic dielectric constant is defined as the real part
        of dielectric function at w=0.

        Parameters:

        eM_LFC: float
            Dielectric constant without local field correction. (RPA, ALDA)
        eM2_NLFC: float
            Dielectric constant with local field correction.
        """
        df = self._new_dielectric_function(xc=xc, q_v=q_v, filename=None,
                                           direction=direction)

        self.context.print('', flush=False)
        self.context.print('%s Macroscopic Dielectric Constant:' % xc)
        self.context.print('  %s direction' % direction, flush=False)
        self.context.print('    Without local field: %f' % df.eps0,
                           flush=False)
        self.context.print('    Include local field: %f' % df.eps)

        return df.eps0, df.eps

    def _new_eels_spectrum(self, xc='RPA', q_c=[0, 0, 0],
                           direction='x', filename='eels.csv'):
        chi = self._new_chi(xc=xc, q_c=q_c, direction=direction)
        eels = chi.eels_spectrum()
        eels.write(filename)
        return eels

    def get_eels_spectrum(self, *args, **kwargs):
        """..."""
        return self._new_eels_spectrum(*args, **kwargs).unpack()

    def _new_polarizability(self, xc='RPA', direction='x', q_c=[0, 0, 0],
                            filename='polarizability.csv'):
        r"""Calculate the polarizability alpha.
        In 3D the imaginary part of the polarizability is related to the
        dielectric function by Im(eps_M) = 4 pi * Im(alpha). In systems
        with reduced dimensionality the converged value of alpha is
        independent of the cell volume. This is not the case for eps_M,
        which is ill-defined. A truncated Coulomb kernel will always give
        eps_M = 1.0, whereas the polarizability maintains its structure.

        By default, generate a file 'polarizability.csv'. The five columns are:
        frequency (eV), Real(alpha0), Imag(alpha0), Real(alpha), Imag(alpha)
        alpha0 is the result without local field effects and the
        dimension of alpha is \AA to the power of non-periodic directions
        """

        # gs: ResponseGroundStateAdapter from gpaw.response.groundstate
        # gd: GridDescriptor object from gpaw.grid_descriptor
        cell_cv = self.gs.gd.cell_cv

        # pbc_c: np.ndarray of type bool. Describes periodic directions.
        pbc_c = self.gs.pbc

        if pbc_c.all():
            V = 1.0
        else:
            V = np.abs(np.linalg.det(cell_cv[~pbc_c][:, ~pbc_c]))

        if not self.coulomb.truncation:
            """Standard expression for the polarizability"""
            df = self._new_dielectric_function(xc=xc, q_c=q_c,
                                               filename=None,
                                               direction=direction)

            df0_w = df.df_NLFC_w
            df_w = df.df_LFC_w
            alpha_w = V * (df_w - 1.0) / (4 * pi)
            alpha0_w = V * (df0_w - 1.0) / (4 * pi)
        else:
            # Since eps_M = 1.0 for a truncated Coulomb interaction, it does
            # not make sense to apply it here. Instead one should define the
            # polarizability by
            #
            #     alpha * eps_M^{-1} = -L / (4 * pi) * <v_ind>
            #
            # where <v_ind> = 4 * pi * \chi / q^2 is the averaged induced
            # potential (relative to the strength of the  external potential).
            # With the bare Coulomb potential, this expression is equivalent to
            # the standard one. In a 2D system \chi should be calculated with a
            # truncated Coulomb potential and eps_M = 1.0

            self.context.print('Using truncated Coulomb interaction')
            chi = self._new_chi(xc=xc, q_c=q_c, direction=direction)

            alpha_w = -V / (4 * pi) * chi.chi_wGG[:, 0, 0]
            alpha0_w = -V / (4 * pi) * chi.chi0_wGG[:, 0, 0]

            alpha_w = self.collect(alpha_w)
            alpha0_w = self.collect(alpha0_w)

        # Convert to external units
        hypervol = Bohr**sum(~pbc_c)
        alpha0_w *= hypervol
        alpha_w *= hypervol

        pol = Polarizability(self.context, self.wd, alpha0_w, alpha_w)
        pol.write(filename)
        return pol

    def get_polarizability(self, *args, **kwargs):
        return self._new_polarizability(*args, **kwargs).unpack()


class DielectricFunction(DielectricFunctionCalculator):
    """This class defines dielectric function related physical quantities."""

    def __init__(self, calc, *,
                 frequencies=None,
                 ecut=50,
                 hilbert=True,
                 nbands=None, eta=0.2,
                 intraband=True, nblocks=1, world=mpi.world, txt=sys.stdout,
                 truncation=None, disable_point_group=False,
                 disable_time_reversal=False,
                 integrationmode=None, rate=0.0,
                 eshift=0.0):
        """Creates a DielectricFunction object.

        calc: str
            The ground-state calculation file that the linear response
            calculation is based on.
        frequencies:
            Input parameters for frequency_grid.
            Can be an array of frequencies to evaluate the response function at
            or dictionary of parameters for build-in nonlinear grid
            (see :ref:`frequency grid`).
        ecut: float
            Plane-wave cut-off.
        hilbert: bool
            Use hilbert transform.
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
        eshift: float
            Shift unoccupied bands
        """
        gs, context = get_gs_and_context(calc, txt, world, timer=None)
        wd = get_frequency_descriptor(frequencies, gs=gs, nbands=nbands)

        chi0calc = Chi0Calculator(
            gs, context, nblocks=nblocks,
            wd=wd,
            ecut=ecut, nbands=nbands, eta=eta,
            hilbert=hilbert,
            intraband=intraband,
            disable_point_group=disable_point_group,
            disable_time_reversal=disable_time_reversal,
            integrationmode=integrationmode,
            rate=rate, eshift=eshift
        )

        super().__init__(wd=wd, chi0calc=chi0calc, truncation=truncation)


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
