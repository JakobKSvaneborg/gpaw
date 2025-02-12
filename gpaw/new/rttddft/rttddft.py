from __future__ import annotations

from typing import Generator, NamedTuple

import numpy as np

from ase.units import Bohr, Hartree

from gpaw.core.atom_centered_functions import UGAtomCenteredFunctions
from gpaw.external import ExternalPotential, ConstantElectricField
from gpaw.mpi import world
from gpaw.new.ase_interface import ASECalculator
from gpaw.new.calculation import DFTState, DFTCalculation
from gpaw.new.fd.hamiltonian import FDHamiltonian, FDKickHamiltonian
from gpaw.new.fd.pot_calc import FDPotentialCalculator
from gpaw.new.gpw import read_gpw
from gpaw.new.hamiltonian import Hamiltonian
from gpaw.new.lcao.hamiltonian import (HamiltonianMatrixCalculator,
                                       LCAOKickHamiltonian,
                                       LCAOHamiltonian)
from gpaw.new.lcao.ibzwfs import LCAOIBZWaveFunctions
from gpaw.new.lcao.wave_functions import LCAOWaveFunctions
from gpaw.new.pot_calc import PotentialCalculator
from gpaw.new.pw.hamiltonian import PWHamiltonian
from gpaw.new.pwfd.ibzwfs import PWFDIBZWaveFunctions
from gpaw.new.rttddft.td_algorithm import (TDAlgorithm, ECNAlgorithm,
                                           WaveFunctionPropagator,
                                           LCAONumpyPropagator,
                                           FDNumpyPropagator)
from gpaw.new.symmetry import Symmetries
from gpaw.new.wave_functions import WaveFunctions
from gpaw.tddft.units import (asetime_to_autime,
                              autime_to_asetime, au_to_eA)
from gpaw.typing import Vector
from gpaw.utilities.timing import nulltimer


class RTTDDFTHistory:

    kick_strength: Vector | None  # Kick strength in atomic units
    niter: int  # Number of propagation steps
    time: float  # Simulation time in atomic units

    def __init__(self):
        """Object that keeps track of the RT-TDDFT history, that is

        - Has a kick been performed?
        - The number of propagation states performed
        """
        self.kick_strength = None
        self.niter = 0
        self.time = 0.0

    def absorption_kick(self,
                        kick_strength: Vector):
        """ Store the kick strength in history

        At most one kick can be done, and it must happen before any
        propagation steps

        Parameters
        ----------
        kick_strength
            Strength of the kick in atomic units
        """
        assert self.niter == 0, 'Cannot kick if already propagated'
        assert self.kick_strength is None, 'Cannot kick if already kicked'
        self.kick_strength = np.array(kick_strength, dtype=float)

    def propagate(self,
                  time_step: float) -> float:
        """ Increment the number of propagation steps and simulation time
        in history

        Parameters
        ----------
        time_step
            Time step in atomic units

        Returns
        -------
        The new simulation time in atomic units
        """
        self.niter += 1
        self.time += time_step

        return self.time

    def todict(self):
        absorption_kick = self.absorption_kick
        if absorption_kick is not None:
            absorption_kick = absorption_kick.tolist()
        return {'niter': self.niter, 'time': self.time,
                'absorption_kick': absorption_kick}


class RTTDDFTResult(NamedTuple):

    """ Results are stored in atomic units, but displayed to the user in
    ASE units
    """

    time: float  # Time in atomic units
    dipolemoment: Vector  # Dipole moment in atomic units
    norm: float  # Integral of density

    def __repr__(self):
        timestr = f'{self.time * autime_to_asetime:.3f} Å√(u/eV)'
        dmstr = ', '.join([f'{dm * au_to_eA:10.4g}'
                           for dm in self.dipolemoment])
        dmstr = f'[{dmstr}]'

        return (f'{self.__class__.__name__}: '
                f'(time: {timestr}, dipolemoment: {dmstr} eÅ)')


class RTTDDFT:
    def __init__(self,
                 state: DFTState,
                 pot_calc: PotentialCalculator,
                 hamiltonian,
                 history: RTTDDFTHistory,
                 td_algorithm: TDAlgorithm | None = None):
        if td_algorithm is None:
            td_algorithm = ECNAlgorithm()

        # Disable symmetries, ie keep only identity operation
        # I suppose this should be done in the kick, as the kick breaks
        # the symmetry
        cell = state.ibzwfs.ibz.symmetries.cell_cv
        natoms = state.ibzwfs.ibz.symmetries.atommap_sa.shape[1]
        atommaps = np.arange(natoms).reshape((1, natoms))
        state.ibzwfs.ibz.symmetries = Symmetries(cell=cell, atommaps=atommaps)

        self.state = state
        self.pot_calc = pot_calc
        self.td_algorithm = td_algorithm
        self.hamiltonian = hamiltonian
        self.history = history

        self.kick_ext: ExternalPotential | None = None

        if isinstance(hamiltonian, LCAOHamiltonian):
            # self.calculate_dipole_moment = self._calculate_dipole_moment_lcao
            self.calculate_dipole_moment = self._calculate_dipole_moment
            self.mode = 'lcao'
        elif isinstance(hamiltonian, FDHamiltonian):
            self.calculate_dipole_moment = self._calculate_dipole_moment
            self.mode = 'fd'
        elif isinstance(hamiltonian, PWHamiltonian):
            raise NotImplementedError('PW TDDFT is not implemented')
        else:
            raise ValueError(f"I don\'t know {hamiltonian} "
                             f'({type(hamiltonian)})')
        # Dipole moment operators in each Cartesian direction
        # Only usable for LCAO
        # TODO is there even a point in caching these? I don't think it saves
        # much time
        self.dm_operator_c: list[HamiltonianMatrixCalculator] | None = None

        self.timer = nulltimer
        self.log = print

    @classmethod
    def from_dft_calculation(cls,
                             calc: ASECalculator | DFTCalculation,
                             td_algorithm: TDAlgorithm | None = None):

        if isinstance(calc, DFTCalculation):
            dft = calc
        else:
            assert calc.dft is not None
            dft = calc.dft

        state = dft.get_state()
        pot_calc = dft.pot_calc
        hamiltonian = dft.scf_loop.hamiltonian
        history = RTTDDFTHistory()

        return cls(state, pot_calc, hamiltonian, td_algorithm=td_algorithm,
                   history=history)

    @classmethod
    def from_dft_file(cls,
                      filepath: str,
                      td_algorithm: TDAlgorithm | None = None):
        _, dft, params, builder = read_gpw(filepath,
                                           log='-',
                                           comm=world,
                                           force_complex_dtype=True)

        state = dft.get_state()
        pot_calc = dft.pot_calc
        hamiltonian = builder.create_hamiltonian_operator()
        history = RTTDDFTHistory()

        return cls(state, pot_calc, hamiltonian, td_algorithm=td_algorithm,
                   history=history)

    def absorption_kick(self,
                        kick_strength: Vector):
        """Kick with a weak electric field.

        Parameters
        ----------
        kick_strength
            Strength of the kick in atomic units
        """
        with self.timer('Kick'):
            kick_strength = np.array(kick_strength, dtype=float)
            self.history.absorption_kick(kick_strength)

            magnitude = np.sqrt(np.sum(kick_strength**2))
            direction = kick_strength / magnitude
            dirstr = [f'{d:.4f}' for d in direction]

            self.log('----  Applying absorption kick')
            self.log(f'----  Magnitude: {magnitude:.8f} Hartree/Bohr')
            self.log(f'----  Direction: {dirstr}')

            # Create Hamiltonian object for absorption kick
            cef = ConstantElectricField(magnitude * Hartree / Bohr, direction)

            kw = dict()
            if self.mode == 'fd':
                kw['nkicks'] = int(round(magnitude / 1.0e-4))
                if kw['nkicks'] < 1:
                    kw['nkicks'] = 1

            # Propagate kick
            return self.kick(cef, **kw)

    def kick(self,
             ext: ExternalPotential,
             nkicks: int = 10):
        """Kick with any external potential.

        Note that unless this function is called by absorption_kick, the kick
        is not logged in history

        Parameters
        ----------
        ext
            External potential
        """
        with self.timer('Kick'):
            assert isinstance(self.state.density.nct_aX,
                              UGAtomCenteredFunctions)
            self.log('----  Applying kick')
            self.log(f'----  {ext}')
            self.kick_ext = ext

            # For the kick, the propagator is always ECN
            td_algorithm = ECNAlgorithm()
            wf_propagator = self.kick_propagator(ext)

            assert isinstance(self.pot_calc, FDPotentialCalculator)
            for l in range(nkicks):
                td_algorithm.propagate_wfs(1 / nkicks,
                                           state=self.state,
                                           pot_calc=self.pot_calc,
                                           wf_propagator=wf_propagator)
            td_algorithm.update_time_dependent_operators(self.state,
                                                         self.pot_calc)

            dipolemoment_xv = [
                self.calculate_dipole_moment(wfs)  # type: ignore
                for wfs in self.state.ibzwfs]
            dipolemoment_v = np.sum(dipolemoment_xv, axis=0)
            norm = np.sum(self.state.density.nct_aX.integrals)
            result = RTTDDFTResult(time=0,
                                   dipolemoment=dipolemoment_v,
                                   norm=norm)
            return result

    @property
    def _wf_propagator_class(self) -> type[WaveFunctionPropagator]:
        cls: type[WaveFunctionPropagator]
        if self.mode == 'lcao':
            cls = LCAONumpyPropagator
        elif self.mode == 'fd':
            cls = FDNumpyPropagator
        else:
            raise RuntimeError(f'Mode {self.mode} is unexpected')
        return cls

    def wf_propagator(self) -> WaveFunctionPropagator:
        """ Wave function propagator

        Corresponding to the mode and type of parallelization
        """
        return self._wf_propagator_class(self.hamiltonian, self.state)

    def kick_propagator(self,
                        ext: ExternalPotential) -> WaveFunctionPropagator:
        """ Wave function propagator

        Corresponding to the mode and type of parallelization
        """
        kick_hamiltonian: Hamiltonian
        assert isinstance(self.pot_calc, FDPotentialCalculator)
        if self.mode == 'lcao':
            assert isinstance(self.state.ibzwfs, LCAOIBZWaveFunctions)
            kick_hamiltonian = LCAOKickHamiltonian(self.hamiltonian.basis,
                                                   self.state.ibzwfs,
                                                   ext,
                                                   self.pot_calc)
        elif self.mode == 'fd':
            assert isinstance(self.state.ibzwfs, PWFDIBZWaveFunctions)
            kwargs = dict(kin_stencil=len(self.hamiltonian.kin.coef_p),
                          xp=self.hamiltonian.kin.xp)
            layout = self.state.potential.dH_asii.layout
            kick_hamiltonian = FDKickHamiltonian(self.hamiltonian.grid,
                                                 ext,
                                                 self.state.ibzwfs,
                                                 self.pot_calc,
                                                 layout,
                                                 **kwargs)

        else:
            raise RuntimeError(f'Mode {self.mode} is unexpected')
        return self._wf_propagator_class(kick_hamiltonian, self.state)

    def ipropagate(self,
                   time_step: float = 10.0,
                   maxiter: int = 2000,
                   ) -> Generator[RTTDDFTResult, None, None]:
        """Propagate the electronic system.

        Parameters
        ----------
        time_step
            Time step in ASE time units Å√(u/eV)
        iterations
            Number of propagation steps
        """

        assert isinstance(self.state.density.nct_aX, UGAtomCenteredFunctions)
        time_step = time_step * asetime_to_autime

        for iteration in range(maxiter):
            self.td_algorithm.propagate(time_step,
                                        state=self.state,
                                        pot_calc=self.pot_calc,
                                        wf_propagator=self.wf_propagator())
            time = self.history.propagate(time_step)
            dipolemoment_xv = [
                self.calculate_dipole_moment(wfs)  # type: ignore
                for wfs in self.state.ibzwfs]
            dipolemoment_v = np.sum(dipolemoment_xv, axis=0)
            norm = np.sum(self.state.density.nct_aX.integrals)
            result = RTTDDFTResult(time=time,
                                   dipolemoment=dipolemoment_v,
                                   norm=norm)
            yield result

    def _calculate_dipole_moment(self, wfs: WaveFunctions) -> np.ndarray:
        dipolemoment_v = self.state.density.calculate_dipole_moment(
            self.pot_calc.relpos_ac)

        return dipolemoment_v

    def _calculate_dipole_moment_lcao(self,
                                      wfs: LCAOWaveFunctions) -> np.ndarray:
        """ Calculates the dipole moment

        The dipole moment is calculated as the expectation value of the
        dipole moment operator, i.e. the trace of it times the density matrix::

          d = - Σ  ρ   d
                μν  μν  νμ

        """
        assert isinstance(wfs, LCAOWaveFunctions)
        if self.dm_operator_c is None:
            self.dm_operator_c = []

            # Create external potentials in each direction
            ext_c = [ConstantElectricField(Hartree / Bohr, dir)
                     for dir in np.eye(3)]
            dm_operator_c = [self.hamiltonian.create_kick_matrix_calculator(
                self.state, ext, self.pot_calc) for ext in ext_c]
            self.dm_operator_c = dm_operator_c

        dm_v = np.zeros(3)
        for c, dm_operator in enumerate(self.dm_operator_c):
            rho_MM = wfs.calculate_density_matrix()
            dm_MM = dm_operator.calculate_matrix(wfs)
            dm = - np.einsum('MN,NM->', rho_MM, dm_MM.data)
            assert np.abs(dm.imag) < 1e-20
            dm_v[c] = dm.real

        return dm_v
