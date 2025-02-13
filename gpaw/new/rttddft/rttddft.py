from __future__ import annotations

from typing import Generator, NamedTuple

import numpy as np

from ase import Atoms
from ase.units import Bohr, Hartree
from ase.io.ulm import Reader

from gpaw.core.atom_centered_functions import UGAtomCenteredFunctions
from gpaw.external import ExternalPotential, ConstantElectricField
from gpaw.mpi import broadcast, world
from gpaw.new.ase_interface import ASECalculator
from gpaw.new.calculation import DFTState
from gpaw.new.fd.hamiltonian import FDHamiltonian, FDKickHamiltonian
from gpaw.new.fd.pot_calc import FDPotentialCalculator
from gpaw.new.gpw import read_gpw
from gpaw.new.rttddft.gpw import read_rttddft, write_rttddft
from gpaw.new.hamiltonian import Hamiltonian
from gpaw.new.input_parameters import InputParameters
from gpaw.new.lcao.hamiltonian import (HamiltonianMatrixCalculator,
                                       LCAOKickHamiltonian,
                                       LCAOHamiltonian)
from gpaw.new.lcao.ibzwfs import LCAOIBZWaveFunctions
from gpaw.new.lcao.wave_functions import LCAOWaveFunctions
from gpaw.new.pot_calc import PotentialCalculator
from gpaw.new.pw.hamiltonian import PWHamiltonian
from gpaw.new.pwfd.ibzwfs import PWFDIBZWaveFunctions
from gpaw.new.rttddft.td_algorithm import create_td_algorithm, TDAlgorithmLike
from gpaw.new.rttddft.history import RTTDDFTHistory
from gpaw.new.wave_functions import WaveFunctions
from gpaw.tddft.units import (asetime_to_autime,
                              autime_to_asetime, au_to_eA)
from gpaw.typing import Vector
from gpaw.utilities.timing import nulltimer


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
                 td_algorithm: TDAlgorithmLike = None,
                 *,
                 dft_params: InputParameters):
        if len(state.ibzwfs.ibz.symmetries.op_scc) > 1:
            raise ValueError('Symmetries are not allowed for TDDFT. '
                             'Run the ground state calculation with '
                             'symmetry={"point_group": False}.')

        self.state = state
        self.pot_calc = pot_calc
        self.td_algorithm = create_td_algorithm(td_algorithm)
        self.hamiltonian = hamiltonian
        self.history = history
        self.dft_params = dft_params

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

    @property
    def atoms(self) -> Atoms:
        """ Get ASE atoms object """
        grid = self.state.density.grid
        symbols = [setup.symbol for setup in self.pot_calc.setups]
        cell_cv = grid.cell_cv * Bohr
        positions_av = self.pot_calc.relpos_ac @ cell_cv
        pbc_c = grid.pbc_c
        return Atoms(symbols, positions_av, cell=cell_cv, pbc=pbc_c)

    @property
    def td_params(self):
        params = {'td_algorithm': self.td_algorithm.todict()}
        return params

    @classmethod
    def from_dft_calculation(cls,
                             calc: ASECalculator,
                             td_algorithm: TDAlgorithmLike = None):

        assert calc.dft is not None
        dft = calc.dft

        state = dft.get_state()
        pot_calc = dft.pot_calc
        hamiltonian = dft.scf_loop.hamiltonian
        history = RTTDDFTHistory()

        return cls(state, pot_calc, hamiltonian,
                   history=history, dft_params=calc.params,
                   td_algorithm=td_algorithm)

    @classmethod
    def from_dft_file(cls,
                      filepath: str,
                      td_algorithm: TDAlgorithmLike = None):
        _, dft, params, builder = read_gpw(filepath,
                                           log='-',
                                           comm=world,
                                           force_complex_dtype=True)

        state = dft.get_state()
        pot_calc = dft.pot_calc
        hamiltonian = builder.create_hamiltonian_operator()
        history = RTTDDFTHistory()

        return cls(state, pot_calc, hamiltonian,
                   history=history, dft_params=params,
                   td_algorithm=td_algorithm)

    @classmethod
    def from_rttddft_file(cls,
                          filepath: str):
        _, state, history, dft_params, params, builder = read_rttddft(
            filepath, log='-', comm=world)

        pot_calc = builder.create_potential_calculator()
        hamiltonian = builder.create_hamiltonian_operator()
        history = RTTDDFTHistory()

        return cls(state, pot_calc, hamiltonian,
                   history=history, dft_params=dft_params, **params)

    @classmethod
    def from_file(cls,
                  filepath: str,
                  **kwargs):
        if world.rank == 0:
            with Reader(filepath) as reader:
                tag = reader.get_tag()
                broadcast(tag, comm=world)
        else:
            tag = broadcast(None, comm=world)
        tag = tag.lower()

        if tag == 'gpaw':
            return cls.from_dft_file(filepath, **kwargs)
        if tag == 'gpaw-rttddft':
            return cls.from_rttddft_file(filepath, **kwargs)

        raise ValueError(f'Unknown file. Tag {tag}')

    def write(self,
              filename: str):
        write_rttddft(filename,
                      self.atoms,
                      self.dft_params,
                      self.td_params,
                      self.state,
                      self.history)

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
            td_algorithm = create_td_algorithm('ecn')
            hamiltonian = self.kick_hamiltonian(ext)

            assert isinstance(self.pot_calc, FDPotentialCalculator)
            for l in range(nkicks):
                td_algorithm.propagate_wfs(1 / nkicks,
                                           state=self.state,
                                           pot_calc=self.pot_calc,
                                           hamiltonian=hamiltonian)
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

    def kick_hamiltonian(self,
                         ext: ExternalPotential) -> Hamiltonian:
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
        return kick_hamiltonian

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
                                        hamiltonian=self.hamiltonian)
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
