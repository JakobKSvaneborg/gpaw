from __future__ import annotations

from collections.abc import Generator
from typing import NamedTuple

import numpy as np
from ase import Atoms
from ase.io.ulm import Reader
from ase.units import Bohr, Hartree

from gpaw.dft import Parameters
from gpaw.external import ConstantElectricField, ExternalPotential
from gpaw.mpi import broadcast, normalize_communicator
from gpaw.new.ase_interface import ASECalculator
from gpaw.new.fd.hamiltonian import FDHamiltonian, FDKickHamiltonian
from gpaw.new.fd.pot_calc import FDPotentialCalculator
from gpaw.new.gpw import read_gpw
from gpaw.new.hamiltonian import Hamiltonian
from gpaw.new.lcao.hamiltonian import LCAOHamiltonian, LCAOKickHamiltonian
from gpaw.new.lcao.ibzwfs import LCAOIBZWaveFunctions
from gpaw.new.pot_calc import PotentialCalculator
from gpaw.new.pw.hamiltonian import PWHamiltonian
from gpaw.new.pwfd.ibzwfs import PWFDIBZWaveFunctions
from gpaw.new.rttddft.gpw import read_rttddft, write_rttddft
from gpaw.new.rttddft.history import RTTDDFTHistory
from gpaw.new.rttddft.state import RTTDDFTState
from gpaw.new.rttddft.td_algorithm import TDAlgorithmLike, create_td_algorithm
from gpaw.tddft.units import asetime_to_autime, au_to_eA, autime_to_asetime
from gpaw.typing import Vector
from gpaw.utilities import reconstruct_atoms
from gpaw.utilities.timing import nulltimer


class RTTDDFTResult(NamedTuple):

    """ Results are stored in atomic units, but displayed to the user in
    ASE units.
    """

    time: float  # Time in atomic units
    dipolemoment: Vector  # Dipole moment in atomic units

    def __repr__(self):
        timestr = f'{self.time * autime_to_asetime:.3f} Å√(u/eV)'
        dmstr = ', '.join([f'{dm * au_to_eA:10.4g}'
                           for dm in self.dipolemoment])
        dmstr = f'[{dmstr}]'

        return (f'{self.__class__.__name__}: '
                f'(time: {timestr}, dipolemoment: {dmstr} eÅ)')

    @classmethod
    def from_state(cls,
                   time: float,
                   state: RTTDDFTState,
                   pot_calc: PotentialCalculator) -> RTTDDFTResult:
        relpos_ac = pot_calc.relpos_ac
        dipolemoment = state.density.calculate_dipole_moment(relpos_ac)

        return RTTDDFTResult(time=time, dipolemoment=dipolemoment)


class RTTDDFT:

    """Real-time time-propagation TDDFT calculator.


    Parameters
    ----------
    state
        State containing wave functions and potentials.
    pot_calc
        Potential calculator.
    hamiltonian
        Time-dependent Hamiltonian object.
    history
        History object.
    td_algorithm
        Propagation algorithm for the state.
    dft_params
        Parameters used in underlying DFT calculation.
    """
    def __init__(self,
                 state: RTTDDFTState,
                 pot_calc: PotentialCalculator,
                 hamiltonian,
                 history: RTTDDFTHistory,
                 td_algorithm: TDAlgorithmLike = None,
                 *,
                 dft_params: Parameters,
                 world=None):
        world = normalize_communicator(world)
        if world.size > 1:
            raise NotImplementedError('Parallel execution not implemented')

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
            self.mode = 'lcao'
        elif isinstance(hamiltonian, FDHamiltonian):
            self.mode = 'fd'
        elif isinstance(hamiltonian, PWHamiltonian):
            raise NotImplementedError('PW TDDFT is not implemented')
        else:
            raise ValueError(f"I don't know {hamiltonian} "
                             f'({type(hamiltonian)})')

        self.timer = nulltimer
        self.log = print

    @property
    def atoms(self) -> Atoms:
        """ Get ASE atoms object. """
        grid = self.state.density.grid
        return reconstruct_atoms(grid, self.pot_calc.setups,
                                 self.pot_calc.relpos_ac)

    @property
    def td_params(self):
        params = {'td_algorithm': self.td_algorithm.todict()}
        return params

    @classmethod
    def from_dft_calculation(cls,
                             calc: ASECalculator,
                             td_algorithm: TDAlgorithmLike = None):
        """ Set up the RTTDDFT object from a DFT calculation file.

        Parameters
        ----------
        filepath
            Filename of the DFT calculation file.
        td_algorithm
            Propagation algorithm for the state.
        """

        assert calc.dft is not None
        dft = calc.dft

        state = RTTDDFTState(dft.ibzwfs, dft.density,
                             dft.potential, dft.energies)
        pot_calc = dft.pot_calc
        hamiltonian = dft.scf_loop.hamiltonian
        history = RTTDDFTHistory()

        return cls(state, pot_calc, hamiltonian,
                   history=history, dft_params=calc.params,
                   td_algorithm=td_algorithm)

    @classmethod
    def from_dft_file(cls,
                      filepath: str,
                      td_algorithm: TDAlgorithmLike = None,
                      world=None):
        """ Set up the RTTDDFT object from a DFT calculation file.

        Parameters
        ----------
        filepath
            Filename of the DFT calculation file.
        td_algorithm
            Propagation algorithm for the state.
        """
        world = normalize_communicator(world)
        _, dft, builder = read_gpw(filepath,
                                   log='-',
                                   comm=world,
                                   force_complex_dtype=True)

        state = RTTDDFTState(dft.ibzwfs, dft.density,
                             dft.potential, dft.energies)
        pot_calc = dft.pot_calc
        hamiltonian = builder.create_hamiltonian_operator()
        history = RTTDDFTHistory()

        return cls(state, pot_calc, hamiltonian,
                   history=history, dft_params=dft.params,
                   td_algorithm=td_algorithm)

    @classmethod
    def from_rttddft_file(cls,
                          filepath: str,
                          world=None):
        """ Set up the RTTDDFT object from a restart file.

        Parameters
        ----------
        filepath
            Filename of the restart file.
        """
        world = normalize_communicator(world)
        _, state, history, dft_params, params, builder = read_rttddft(
            filepath, log='-', comm=world)

        pot_calc = builder.create_potential_calculator()
        hamiltonian = builder.create_hamiltonian_operator()

        return cls(state, pot_calc, hamiltonian,
                   history=history, dft_params=dft_params, **params)

    @classmethod
    def from_file(cls,
                  filepath: str,
                  *,
                  world=None,
                  **kwargs):
        """ Set up the RTTDDFT object from a file.

        The file can be a DFT calculation file or a RTTDDFT restart file.
        This is inferred from the file contents.
        See :meth:`~from_dft_file` and :meth:`~from_rttddft_file`.

        Parameters
        ----------
        filepath
            Filename.
        kwargs
            Parameters passed to the :meth:`~from_dft_file` if
            `filepath` is a DFT calculation file. No parameters
            are allowed for RTTDDFT restart files.
        """
        world = normalize_communicator(world)

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
            if kwargs.pop('td_algorithm', None) is not None:
                raise ValueError('Parameter td_algorithm may not be '
                                 'given when restarting.')
            return cls.from_rttddft_file(filepath, **kwargs)

        raise ValueError(f'Unknown file. Tag {tag}')

    def write(self,
              filename: str):
        """ Write a restart file.

        Parameters
        ----------
        filename
            Filename of the restart file.
        """
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
            Strength of the kick in atomic units.
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
                # Different behavior between LCAO and FD. See #1423
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
        is not logged in history.

        Parameters
        ----------
        ext
            External potential.
        nkicks
            Propagate the wave functions nkicks times, using a kick
            that is scaled down by 1/nkicks. Propagating the kick in
            several steps improves the accuracy of the propagator.
        """
        # Construct the kick hamiltonian
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

        with self.timer('Kick'):
            self.log('----  Applying kick')
            self.log(f'----  {ext}')
            self.kick_ext = ext

            # For the kick, the propagator is always ECN
            td_algorithm = create_td_algorithm('ecn')

            for l in range(nkicks):
                td_algorithm.propagate_wfs(1 / nkicks,
                                           state=self.state,
                                           hamiltonian=kick_hamiltonian)
            td_algorithm.update_time_dependent_operators(self.state,
                                                         self.pot_calc)

            return RTTDDFTResult.from_state(time=self.history.time,
                                            pot_calc=self.pot_calc,
                                            state=self.state)

        return kick_hamiltonian

    def ipropagate(self,
                   time_step: float = 1e-3,
                   maxiter: int = 2000,
                   ) -> Generator[RTTDDFTResult]:
        """Propagate the electronic system.

        Parameters
        ----------
        time_step
            Time step in ASE time units Å√(u/eV).
        maxiter
            Number of propagation steps.
        """

        time_step = time_step * asetime_to_autime

        for iteration in range(maxiter):
            self.td_algorithm.propagate(time_step,
                                        state=self.state,
                                        pot_calc=self.pot_calc,
                                        hamiltonian=self.hamiltonian)
            time = self.history.propagate(time_step)

            yield RTTDDFTResult.from_state(time=time,
                                           pot_calc=self.pot_calc,
                                           state=self.state)
