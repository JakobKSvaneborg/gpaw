from __future__ import annotations

import warnings
from time import ctime
from typing import Union, TYPE_CHECKING

import numpy as np
from ase.units import Ha

import gpaw
import gpaw.mpi as mpi
from gpaw.response.chi0_data import (Chi0Data, Chi0BodyData,
                                     Chi0OpticalExtensionData)
from gpaw.response.frequencies import FrequencyDescriptor
from gpaw.response.pair_functions import SingleQPWDescriptor
from gpaw.response.hilbert import HilbertTransform
from gpaw.response import timer
from gpaw.response.pair import KPointPairFactory
from gpaw.response.pw_parallelization import PlaneWaveBlockDistributor
from gpaw.typing import Array1D
from gpaw.utilities.memory import maxrss
from gpaw.response.chi0_base import Chi0ComponentPWCalculator, Chi0Integrand
from gpaw.response.integrators import (
    HermitianOpticalLimit, HilbertOpticalLimit, OpticalLimit,
    HilbertOpticalLimitTetrahedron,
    Hermitian, Hilbert, HilbertTetrahedron, GenericUpdate)

if TYPE_CHECKING:
    from gpaw.response.context import ResponseContext
    from gpaw.response.groundstate import ResponseGroundStateAdapter
    from gpaw.response.pair import ActualPairDensityCalculator


def find_maximum_frequency(kpt_u: list,
                           context: ResponseContext,
                           nbands=0) -> float:
    """Determine the maximum electron-hole pair transition energy."""
    epsmin = 10000.0
    epsmax = -10000.0

    # kpt_u: list of KPoint from gpaw.kpoint
    for kpt in kpt_u:
        epsmin = min(epsmin, kpt.eps_n[0])
        epsmax = max(epsmax, kpt.eps_n[nbands - 1])

    context.print('Minimum eigenvalue: %10.3f eV' % (epsmin * Ha),
                  flush=False)
    context.print('Maximum eigenvalue: %10.3f eV' % (epsmax * Ha))

    return epsmax - epsmin


class Chi0Calculator:
    def __init__(self, kptpair_factory: KPointPairFactory,
                 context: ResponseContext | None = None,
                 eshift=0.0,
                 intraband=True,
                 rate=0.0,
                 **kwargs):
        self.kptpair_factory = kptpair_factory

        # gs: ResponseGroundStateAdapter from gpaw.response.groundstate
        self.gs = kptpair_factory.gs

        # context: ResponseContext from gpaw.response.context
        if context is None:
            context = kptpair_factory.context
        self.context = context

        self.chi0_body_calc = Chi0BodyCalculator(
            kptpair_factory, context=context,
            eshift=eshift, **kwargs)
        self.chi0_opt_ext_calc = Chi0OpticalExtensionCalculator(
            kptpair_factory, context=context,
            intraband=intraband, rate=rate, **kwargs)

    @property
    def pair_calc(self) -> ActualPairDensityCalculator:
        # In a future refactor, we should find better ways to access the pair
        # density calculator (and the pair density paw corrections) XXX

        # pair_calc: ActualPairDensityCalculator from gpaw.response.pair
        return self.chi0_body_calc.pair_calc

    def create_chi0(self, q_c: list | np.ndarray) -> Chi0Data:

        # chi0_body: Chi0BodyData from gpaw.response.chi0_data
        chi0_body = self.chi0_body_calc.create_chi0_body(q_c)

        # chi0: Chi0Data from gpaw.response.chi0_data
        chi0 = Chi0Data.from_chi0_body_data(chi0_body)
        return chi0

    def calculate(self, q_c: list | np.ndarray) -> Chi0Data:
        """Calculate chi0 (possibly with optical extensions).

        Parameters
        ----------
        q_c : list or ndarray
            Momentum vector.

        Returns
        -------
        chi0 : Chi0Data
            Data object containing the chi0 data arrays along with basis
            representation descriptors and blocks distribution
        """
        # Calculate body

        # chi0_body: Chi0BodyData from gpaw.response.chi0_data
        chi0_body = self.chi0_body_calc.calculate(q_c)
        # SingleQPWDescriptor from gpaw.response.pair_functions
        qpd = chi0_body.qpd

        # Calculate optical extension
        if qpd.optical_limit:
            if not abs(self.chi0_body_calc.eshift) < 1e-8:
                raise NotImplementedError("No wings eshift available")
            chi0_opt_ext = self.chi0_opt_ext_calc.calculate(qpd=qpd)
        else:
            chi0_opt_ext = None

        self.context.print('\nFinished calculating chi0\n')

        return Chi0Data(chi0_body, chi0_opt_ext)

    @timer('Calculate CHI_0')
    def update_chi0(self,
                    chi0: Chi0Data,
                    m1: int, m2: int, spins: list) -> Chi0Data:
        """In-place calculation of chi0 (with optical extension).

        Parameters
        ----------
        chi0 : Chi0Data
            Data and representation object
        m1 : int
            Lower band cutoff for band summation
        m2 : int
            Upper band cutoff for band summation
        spins : list
            List of spin indices to include in the calculation

        Returns
        -------
        chi0 : Chi0Data
        """
        self.chi0_body_calc.update_chi0_body(chi0.body, m1, m2, spins)
        if chi0.optical_limit:
            if not abs(self.chi0_body_calc.eshift) < 1e-8:
                raise NotImplementedError("No wings eshift available")
            assert chi0.optical_extension is not None
            # Update the head and wings
            self.chi0_opt_ext_calc.update_chi0_optical_extension(
                chi0.optical_extension, m1, m2, spins)
        return chi0


class Chi0BodyCalculator(Chi0ComponentPWCalculator):
    def __init__(self, *args,
                 eshift=0.0,
                 **kwargs):
        self.eshift = eshift / Ha
        super().__init__(*args, **kwargs)

        # gs: ResponseGroundStateAdapter from gpaw.response.groundstate
        if self.gs.metallic:
            assert abs(self.eshift) < 1e-8, \
                'A rigid energy shift cannot be applied to the conduction '\
                'bands if there is no band gap'

    def create_chi0_body(self, q_c: list | np.ndarray) -> Chi0BodyData:
        # qpd: SingleQPWDescriptor from gpaw.response.pair_functions
        qpd = self.get_pw_descriptor(q_c)
        return self._create_chi0_body(qpd)

    def _create_chi0_body(self, qpd: SingleQPWDescriptor) -> Chi0BodyData:
        return Chi0BodyData(self.wd, qpd, self.get_blockdist())

    def get_blockdist(self) -> PlaneWaveBlockDistributor:
        # integrator: Integrator from gpaw.response.integrators
        #    (or a child of this class)
        return PlaneWaveBlockDistributor(
            self.context.comm,  # _Communicator object from gpaw.mpi
            self.integrator.blockcomm,  # _Communicator object from gpaw.mpi
            self.integrator.kncomm)  # _Communicator object from gpaw.mpi

    def calculate(self, q_c: list | np.ndarray) -> Chi0BodyData:
        """Calculate the chi0 body.

        Parameters
        ----------
        q_c : list or ndarray
            Momentum vector.
        """
        # Construct the output data structure
        # qpd: SingleQPWDescriptor from gpaw.response.pair_functions
        qpd = self.get_pw_descriptor(q_c)
        self.print_info(qpd)
        # chi0_body: Chi0BodyData from gpaw.response.chi0_data
        chi0_body = self._create_chi0_body(qpd)

        # Integrate all transitions into partially filled and empty bands
        m1, m2 = self.get_band_transitions()
        self.update_chi0_body(chi0_body, m1, m2, spins=range(self.gs.nspins))

        return chi0_body

    def update_chi0_body(self,
                         chi0_body: Chi0BodyData,
                         m1: int, m2: int, spins: list | range):
        """In-place calculation of the body.

        Parameters
        ----------
        m1 : int
            Lower band cutoff for band summation
        m2 : int
            Upper band cutoff for band summation
        spins : list
            List of spin indices to include in the calculation
        """
        qpd = chi0_body.qpd

        # Reset PAW correction in case momentum has change
        pairden_paw_corr = self.gs.pair_density_paw_corrections
        self.pawcorr = pairden_paw_corr(chi0_body.qpd)

        self.context.print('Integrating chi0 body.')

        # domain: Domain from from gpaw.response.integrators
        # analyzer: PWSymmetryAnalyzer from gpaw.response.symmetry
        domain, analyzer, prefactor = self.get_integration_domain(qpd, spins)
        integrand = Chi0Integrand(self, qpd=qpd, analyzer=analyzer,
                                  optical=False, m1=m1, m2=m2)

        chi0_body.data_WgG[:] /= prefactor
        if self.hilbert:
            # Allocate a temporary array for the spectral function
            out_WgG = chi0_body.zeros()
        else:
            # Use the preallocated array for direct updates
            out_WgG = chi0_body.data_WgG
        self.integrator.integrate(domain=domain,  # Integration domain
                                  integrand=integrand,
                                  task=self.task,
                                  wd=self.wd,  # Frequency Descriptor
                                  out_wxx=out_WgG)  # Output array

        if self.hilbert:
            # The integrator only returns the spectral function and a Hilbert
            # transform is performed to return the real part of the density
            # response function.
            with self.context.timer('Hilbert transform'):
                # Make Hilbert transform
                ht = HilbertTransform(np.array(self.wd.omega_w), self.eta,
                                      timeordered=self.timeordered)
                ht(out_WgG)
            # Update the actual chi0 array
            chi0_body.data_WgG[:] += out_WgG
        chi0_body.data_WgG[:] *= prefactor

        tmp_chi0_wGG = chi0_body.copy_array_with_distribution('wGG')
        analyzer.symmetrize_wGG(tmp_chi0_wGG)
        chi0_body.data_WgG[:] = chi0_body.blockdist.distribute_as(
            tmp_chi0_wGG, chi0_body.nw, 'WgG')

    def construct_hermitian_task(self):
        return Hermitian(self.integrator.blockcomm, eshift=self.eshift)

    def construct_point_hilbert_task(self):
        return Hilbert(self.integrator.blockcomm, eshift=self.eshift)

    def construct_tetra_hilbert_task(self):
        return HilbertTetrahedron(self.integrator.blockcomm)

    def construct_literal_task(self):
        return GenericUpdate(
            self.eta, self.integrator.blockcomm, eshift=self.eshift)

    def print_info(self, qpd: SingleQPWDescriptor):

        if gpaw.dry_run:
            from gpaw.mpi import SerialCommunicator
            size = gpaw.dry_run
            comm = SerialCommunicator()
            comm.size = size
        else:
            comm = self.context.comm

        q_c = qpd.q_c
        nw = len(self.wd)
        csize = comm.size
        knsize = self.integrator.kncomm.size
        bsize = self.integrator.blockcomm.size
        chisize = nw * qpd.ngmax**2 * 16. / 1024**2 / bsize

        isl = ['', f'{ctime()}',
               'Calculating chi0 body with:',
               self.get_gs_info_string(tab='    '), '',
               '    Linear response parametrization:',
               f'    q_c: [{q_c[0]}, {q_c[1]}, {q_c[2]}]',
               self.get_response_info_string(qpd, tab='    '),
               f'    comm.size: {csize}',
               f'    kncomm.size: {knsize}',
               f'    blockcomm.size: {bsize}']
        if bsize > nw:
            isl.append(
                'WARNING! Your nblocks is larger than number of frequency'
                ' points. Errors might occur, if your submodule does'
                ' not know how to handle this.')
        isl.extend(['',
                    '    Memory estimate of potentially large arrays:',
                    f'        chi0_wGG: {chisize} M / cpu',
                    '        Memory usage before allocation: '
                    f'{(maxrss() / 1024**2)} M / cpu'])
        self.context.print('\n'.join(isl))


class Chi0OpticalExtensionCalculator(Chi0ComponentPWCalculator):

    def __init__(self, *args,
                 intraband=True,
                 rate=0.0,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # In the optical limit of metals, one must add the Drude dielectric
        # response from the free-space plasma frequency of the intraband
        # transitions to the head of the chi0 wings. This is handled by a
        # separate calculator, provided that intraband is set to True.
        if self.gs.metallic and intraband:
            from gpaw.response.chi0_drude import Chi0DrudeCalculator
            if rate == 'eta':
                rate = self.eta * Ha  # external units
            self.rate = rate
            self.drude_calc = Chi0DrudeCalculator(
                self.kptpair_factory,
                disable_point_group=self.disable_point_group,
                disable_time_reversal=self.disable_time_reversal,
                integrationmode=self.integrationmode)
        else:
            self.drude_calc = None
            self.rate = None

    @property
    def nblocks(self) -> int:
        # The optical extensions are not distributed in memory, hence we
        # overwrite nblocks.
        # NB: There can be a mismatch with self.kptpair_factory.nblocks, which
        # seems a bit dangerous XXX
        return 1

    def calculate(self,
                  qpd: SingleQPWDescriptor | None = None
                  ) -> Chi0OpticalExtensionData:
        """Calculate the chi0 head and wings."""
        # Create data object
        if qpd is None:
            qpd = self.get_pw_descriptor(q_c=[0., 0., 0.])

        # wd: FrequencyDescriptor from gpaw.response.frequencies
        chi0_opt_ext = Chi0OpticalExtensionData(self.wd, qpd)

        self.print_info(qpd)

        # Define band transitions
        m1, m2 = self.get_band_transitions()

        # Perform the actual integration
        self.update_chi0_optical_extension(chi0_opt_ext, m1, m2,
                                           spins=range(self.gs.nspins))

        if self.drude_calc is not None:
            # Add intraband contribution
            # drude_calc: Chi0DrudeCalculator from gpaw.response.chi0_drude
            # chi0_drude: Chi0DrudeData from gpaw.response.chi0_data
            chi0_drude = self.drude_calc.calculate(self.wd, self.rate)
            chi0_opt_ext.head_Wvv[:] += chi0_drude.chi_Zvv

        return chi0_opt_ext

    def update_chi0_optical_extension(
            self,
            chi0_optical_extension: Chi0OpticalExtensionData,
            m1: int, m2: int,
            spins: list | range):
        """In-place calculation of the chi0 head and wings.

        Parameters
        ----------
        m1 : int
            Lower band cutoff for band summation
        m2 : int
            Upper band cutoff for band summation
        spins : list
            List of spin indices to include in the calculation
        """
        self.context.print('Integrating chi0 head and wings.')
        chi0_opt_ext = chi0_optical_extension
        qpd = chi0_opt_ext.qpd

        domain, analyzer, prefactor = self.get_integration_domain(qpd, spins)
        integrand = Chi0Integrand(self, qpd=qpd, analyzer=analyzer,
                                  optical=True, m1=m1, m2=m2)

        # We integrate the head and wings together, using the combined index P
        # index v = (x, y, z)
        # index G = (G0, G1, G2, ...)
        # index P = (x, y, z, G1, G2, ...)
        WxvP_shape = list(chi0_opt_ext.WxvG_shape)
        WxvP_shape[-1] += 2
        tmp_chi0_WxvP = np.zeros(WxvP_shape, complex)
        self.integrator.integrate(domain=domain,  # Integration domain
                                  integrand=integrand,
                                  task=self.task,
                                  wd=self.wd,  # Frequency Descriptor
                                  out_wxx=tmp_chi0_WxvP)  # Output array
        if self.hilbert:
            with self.context.timer('Hilbert transform'):
                ht = HilbertTransform(np.array(self.wd.omega_w), self.eta,
                                      timeordered=self.timeordered)
                ht(tmp_chi0_WxvP)
        tmp_chi0_WxvP *= prefactor

        # Fill in wings part of the data, but leave out the head part (G0)
        chi0_opt_ext.wings_WxvG[..., 1:] += tmp_chi0_WxvP[..., 3:]
        analyzer.symmetrize_wxvG(chi0_opt_ext.wings_WxvG)
        # Fill in the head
        chi0_opt_ext.head_Wvv[:] += tmp_chi0_WxvP[:, 0, :3, :3]
        analyzer.symmetrize_wvv(chi0_opt_ext.head_Wvv)

    def construct_hermitian_task(self):
        return HermitianOpticalLimit()

    def construct_point_hilbert_task(self):
        return HilbertOpticalLimit()

    def construct_tetra_hilbert_task(self):
        return HilbertOpticalLimitTetrahedron()

    def construct_literal_task(self):
        return OpticalLimit(eta=self.eta)

    def print_info(self, qpd: SingleQPWDescriptor):
        """Print information about optical extension calculation."""
        isl = ['',
               f'{ctime()}',
               'Calculating chi0 optical extensions with:',
               self.get_gs_info_string(tab='    '),
               '',
               '    Linear response parametrization:',
               self.get_response_info_string(qpd, tab='    ')]
        self.context.print('\n'.join(isl))


class Chi0(Chi0Calculator):
    """Class for calculating non-interacting response functions.
    Tries to be backwards compatible, for now. """

    def __init__(self,
                 calc: str,
                 *,
                 frequencies: Union[dict, Array1D] | None = None,
                 ecut=50,
                 world=mpi.world, txt='-', timer=None,
                 nblocks=1,
                 nbands: int | None = None,
                 domega0: float | None = None,  # deprecated
                 omega2: float | None = None,  # deprecated
                 omegamax: float | None = None,  # deprecated
                 **kwargs):
        """Construct Chi0 object.

        Parameters
        ----------
        calc : str
            The groundstate calculation file that the linear response
            calculation is based on.
        frequencies :
            Input parameters for frequency_grid.
            Can be array of frequencies to evaluate the response function at
            or dictionary of paramaters for build-in nonlinear grid
            (see :ref:`frequency grid`).
        ecut : float
            Energy cutoff.
        hilbert : bool
            Switch for hilbert transform. If True, the full density response
            is determined from a hilbert transform of its spectral function.
            This is typically much faster, but does not work for imaginary
            frequencies.
        nbands : int
            Maximum band index to include.
        timeordered : bool
            Switch for calculating the time ordered density response function.
            In this case the hilbert transform cannot be used.
        eta : float
            Artificial broadening of spectra.
        intraband : bool
            Switch for including the intraband contribution to the density
            response function.
        world : MPI comm instance
            MPI communicator.
        txt : str
            Output file.
        timer : gpaw.utilities.timing.timer instance
        nblocks : int
            Divide the response function into nblocks. Useful when the response
            function is large.
        disable_point_group : bool
            Do not use the point group symmetry operators.
        disable_time_reversal : bool
            Do not use time reversal symmetry.
        integrationmode : str
            Integrator for the kpoint integration.
            If == 'tetrahedron integration' then the kpoint integral is
            performed using the linear tetrahedron method.
        eshift : float
            Shift unoccupied bands
        rate : float,str
            Phenomenological scattering rate to use in optical limit Drude term
            (in eV). If rate='eta', then use input artificial broadening eta as
            rate. Note, for consistency with the formalism the rate is
            implemented as omegap^2 / (omega + 1j * rate)^2 which differ from
            some literature by a factor of 2.


        Attributes
        ----------
        kptpair_factory : gpaw.response.pair.KPointPairFactory instance
            Class for calculating matrix elements of pairs of wavefunctions.

        """
        from gpaw.response.pair import get_gs_and_context

        # gs: ResponseGroundStateAdapter from gpaw.response.groundstate
        # context: ResponseContext from gpaw.response.context
        gs, context = get_gs_and_context(calc, txt, world, timer)

        # bd: BandDescriptor from gpaw.band_descriptor
        nbands = nbands or gs.bd.nbands

        # wd: FrequencyDescriptor from gpaw.response.frequencies
        wd = new_frequency_descriptor(
            gs, context, nbands, frequencies,
            domega0=domega0,
            omega2=omega2, omegamax=omegamax)

        kptpair_factory = KPointPairFactory(gs, context, nblocks=nblocks)

        super().__init__(wd=wd, kptpair_factory=kptpair_factory,
                         nbands=nbands, ecut=ecut, **kwargs)


def new_frequency_descriptor(gs: ResponseGroundStateAdapter,
                             context: ResponseContext,
                             nbands: int,
                             frequencies: None | dict | np.ndarray = None,
                             *, domega0: float | None = None,
                             omega2: float | None = None,
                             omegamax: float | None = None)\
        -> FrequencyDescriptor:

    if domega0 is not None or omega2 is not None or omegamax is not None:
        assert frequencies is None
        frequencies = {'type': 'nonlinear',
                       'domega0': domega0,
                       'omega2': omega2,
                       'omegamax': omegamax}
        warnings.warn(f'Please use frequencies={frequencies}')

    elif frequencies is None:
        frequencies = {'type': 'nonlinear'}

    if (isinstance(frequencies, dict) and
        frequencies.get('omegamax') is None):
        omegamax = find_maximum_frequency(gs.kpt_u, context,
                                          nbands=nbands)
        frequencies['omegamax'] = omegamax * Ha

    wd = FrequencyDescriptor.from_array_or_dict(frequencies)
    return wd
