from __future__ import annotations

from abc import ABC
from functools import partial
from typing import Generator, NamedTuple

import numpy as np
from numpy.linalg import solve

from ase.units import Bohr, Hartree

from gpaw.core.uniform_grid import UGArray, UGDesc
from gpaw.core.atom_arrays import AtomArrays
from gpaw.external import ExternalPotential, ConstantElectricField
from gpaw.typing import Vector
from gpaw.mpi import world
from gpaw.new.ase_interface import ASECalculator
from gpaw.new.calculation import DFTState, DFTCalculation
from gpaw.new.fd.builder import FDHamiltonian, FDKickHamiltonian
from gpaw.new.hamiltonian import Hamiltonian
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from gpaw.new.lcao.hamiltonian import (HamiltonianMatrixCalculator,
                                       LCAOKickHamiltonian,
                                       LCAOHamiltonian)
from gpaw.new.lcao.wave_functions import LCAOWaveFunctions
from gpaw.new.wave_functions import WaveFunctions
from gpaw.new.gpw import read_gpw
from gpaw.new.symmetry import Symmetries
from gpaw.new.pot_calc import PotentialCalculator
from gpaw.new.pw.builder import PWHamiltonian
from gpaw.tddft.solvers.cscg import CSCG
from gpaw.tddft.units import asetime_to_autime, autime_to_asetime, au_to_eA
from gpaw.utilities.timing import nulltimer


class TDAlgorithm:

    def propagate(self,
                  time_step: float,
                  state: DFTState,
                  pot_calc: PotentialCalculator,
                  wf_propagator: WaveFunctionPropagator):
        """ One propagation step, i.e.

        ::

                                 0+
                     ^        -1 /                  ^        -1
          U(0+, 0) = T exp[-iS   | δ(τ) H(r) dτ ] = T exp[-iS  H(r)]
                                 /
                                 0

        (1) Calculate propagator U[H(t)]
        (2) Update wavefunctions ψ_n(t+dt) ← U[H(t)] ψ_n(t)
        (3) Update density and hamiltonian H(t+dt)
        """
        self.propagate_wfs(time_step, state, pot_calc, wf_propagator)
        self.update_time_dependent_operators(state, pot_calc)

    def update_time_dependent_operators(self,
                                        state: DFTState,
                                        pot_calc: PotentialCalculator):
        # Update density
        state.density.update(state.ibzwfs)

        # Calculate Hamiltonian H(t+dt) = H[n[Phi_n]]
        state.potential, _ = pot_calc.calculate(
            state.density, state.ibzwfs, vHt_x=state.potential.vHt_x)

    def propagate_wfs(self,
                      time_step: float,
                      state: DFTState,
                      pot_calc: PotentialCalculator,
                      wf_propagator: WaveFunctionPropagator):
        raise NotImplementedError()

    def get_description(self):
        return self.__class__.__name__


def propagate_wave_functions_numpy(source_C_nM: np.ndarray,
                                   target_C_nM: np.ndarray,
                                   S_MM: np.ndarray,
                                   H_MM: np.ndarray,
                                   dt: float):
    SjH_MM = S_MM + (0.5j * dt) * H_MM
    target_C_nM[:] = source_C_nM @ SjH_MM.conj().T
    target_C_nM[:] = solve(SjH_MM.T, target_C_nM.T).T


class WaveFunctionPropagator(ABC):
    """
    Takes care about propagating wave functions

    Implementations are specific to parallelization scheme (Numpy) and
    type of wave functions (LCAO/FD)
    """

    def __init__(self,
                 hamiltonian: Hamiltonian,
                 state: DFTState):
        raise NotImplementedError

    def propagate(self,
                  wfs: WaveFunctions,
                  time_step: float):
        raise NotImplementedError


class LCAONumpyPropagator(WaveFunctionPropagator):

    def __init__(self,
                 hamiltonian: Hamiltonian,
                 state: DFTState):
        assert isinstance(hamiltonian, LCAOHamiltonian)
        ham_calc = hamiltonian.create_hamiltonian_matrix_calculator(state)
        self.ham_calc = ham_calc

    def propagate(self,
                  wfs: WaveFunctions,
                  time_step: float):
        assert isinstance(wfs, LCAOWaveFunctions)
        H_MM = self.ham_calc.calculate_matrix(wfs)

        # Phi_n <- U[H(t)] Phi_n
        propagate_wave_functions_numpy(wfs.C_nM.data, wfs.C_nM.data,
                                       wfs.S_MM.data,
                                       H_MM.data, time_step)


class FDNumpyPropagator(WaveFunctionPropagator):

    def __init__(self,
                 hamiltonian: Hamiltonian,
                 state: DFTState):
        """
        Parameters
        ----------
        hamiltonian
        """
        assert isinstance(hamiltonian, FDHamiltonian)

        self.timer = nulltimer
        self.preconditioner = None

        self.hamiltonian = hamiltonian
        self.Ht = partial(hamiltonian.apply,
                          state.potential.vt_sR,
                          state.potential.dedtaut_sR,
                          state.ibzwfs, state.density.D_asii)
        if isinstance(hamiltonian, FDKickHamiltonian):
            self.dH = hamiltonian.dH
        else:
            self.dH = state.potential.dH

        wfs = state.ibzwfs.wfs_qs[0][0]
        self.dS = partial(wfs.setups.get_overlap_corrections,
                          wfs.P_ani.layout.atomdist,
                          wfs.xp)

        self.dSinv = wfs.setups.inverse_overlap_correction
        self.layout = wfs.P_ani.layout
        self.pt_aiX = wfs.pt_aiX

    def calculate_projections(self,
                              psit_nR: UGArray,
                              P_ani: AtomArrays | None = None) -> AtomArrays:
        """ Calculate the PAW projections

        Parameters
        ----------
        psit_nR
            Wave functions
        P_ani
            Write the PAW projections here. If None, a new object is allocated

        Returns
        -------
        The projections P_ani
        """
        if P_ani is None:
            P_ani = self.layout.empty(len(psit_nR))
        self.pt_aiX.integrate(psit_nR, P_ani)

        return P_ani

    def propagate(self,
                  wfs: WaveFunctions,
                  time_step: float):
        assert isinstance(wfs, PWFDWaveFunctions)
        assert isinstance(wfs.psit_nX, UGArray)
        psit_nR = wfs.psit_nX

        # Empty arrays
        rhs_nR = psit_nR.new(zeroed=True)
        init_guess_nR = psit_nR.new(zeroed=True)
        hpsit_nR = psit_nR.new(zeroed=True)
        spsit_nR = psit_nR.new(zeroed=True)
        sinvhpsit_nR = psit_nR.new(zeroed=True)

        # Calculate right-hand side of equation
        # ( S + i H dt/2 ) psit(t+dt) = ( S - i H dt/2 ) psit(t)
        self.apply_hamiltonian(psit_nR, hpsit_nR)  # hpsit_nR <- H psit(t)
        self.apply_overlap_operator(psit_nR, spsit_nR)  # spsit_nR <- S psit(t)

        rhs_nR.data[:] = spsit_nR.data - 0.5j * time_step * hpsit_nR.data

        # Calculate (1 - i S^(-1) H dt) psit(t), which is an
        # initial guess for the conjugate gradient solver
        self.apply_inverse_overlap_operator(hpsit_nR, sinvhpsit_nR)
        init_guess_nR.data[:] = (psit_nR.data -
                                 1j * time_step * sinvhpsit_nR.data)

        # Solve A x = b where A is (S + i H dt/2) and b = rhs_kpt.psit_nG
        solver = CSCGAdapter(psit_nR.desc, self)
        solver.solve(init_guess_nR, rhs_nR, psit_nR, time_step)

        self.calculate_projections(psit_nR, wfs.P_ani)

    def dot(self, psit_nR: UGArray, out_nR: UGArray, time_step: float):
        """Applies the propagator matrix to the given wavefunctions.

        (S + i H dt/2 ) psit_nR

        Parameters
        ----------
        psit_nR
            The known wavefunctions
        out_nR
            The result ( S + i H dt/2 ) psit_nR

        """
        self.timer.start('Apply time-dependent operators')

        hpsit_nR = psit_nR.new()
        spsit_nR = psit_nR.new()

        self.apply_hamiltonian(psit_nR, hpsit_nR)
        self.apply_overlap_operator(psit_nR, spsit_nR)
        self.timer.stop('Apply time-dependent operators')

        out_nR.data[:] = spsit_nR.data + 0.5j * time_step * hpsit_nR.data

    def apply_preconditioner(self, psi, psin):
        """Solves preconditioner equation.

        Parameters
        ----------
        psi: List of coarse grids
            the known wavefunctions
        psin: List of coarse grids
            the result

        """
        self.timer.start('Solve TDDFT preconditioner')
        if self.preconditioner is not None:
            self.preconditioner.apply(self.kpt, psi, psin)
        else:
            psin[:] = psi
        self.timer.stop('Solve TDDFT preconditioner')

    def apply_hamiltonian(self,
                          psit_nR: UGArray,
                          out_nR: UGArray,
                          spin: int = 0):
        """
        Apply the hamiltonian on wave functions

        Parameters
        ----------
        psit_nR
            Wave functions
        out_nR
            Result, i.e. hamiltonian acting on wavefunctions
        spin
            Spin
        """
        out_nR.data[:] = 0
        self.Ht(psit_nG=psit_nR, out=out_nR, spin=spin)

        # apply the non-local part for each nucleus
        P_ani = self.calculate_projections(psit_nR)
        P2_ani = P_ani.new()
        self.dH(P_ani, P2_ani, spin)

        # add partial wave pt_nG to psit_nG with proper coefficient
        self.pt_aiX.add_to(out_nR, P2_ani)

    def apply_overlap_operator(self,
                               psit_nR: UGArray,
                               out_nR: UGArray | None = None):
        """
        Apply the overlap operator wave functions on the wave functions:

        ^  ~         ~             ~ a       a   a
        S |ψ (t)〉= |ψ (t)〉+  Σ  |p  (t)〉ΔO   P
            n         n       aij   i        ij  nj

        Parameters
        ----------
        psit_nR
            Wave functions
        out_nR
            Result, i.e. overlap operator acting on wavefunctions
        """

        out_nR.data[:] = psit_nR.data

        P_ani = self.calculate_projections(psit_nR)
        P2_ani = P_ani.new()
        dS_aii = self.dS()
        P_ani.block_diag_multiply(dS_aii, P2_ani)
        self.pt_aiX.add_to(out_nR, P2_ani)

    def apply_inverse_overlap_operator(self,
                                       psit_nR: UGArray,
                                       out_nR: UGArray | None = None):
        """
        Apply the approximate inverse overlap operator on the wave functions:

        ^ -1  ~         ~             ~ a       a   a
        S    |ψ (t)〉= |ψ (t)〉+  Σ  |p  (t)〉ΔC   P
               n         n       aij   i        ij  nj

        Parameters
        ----------
        psit_nR
            Wave functions
        out_nR
            Result, i.e. inverse overlap operator acting on wavefunctions
        """
        out_nR.data[:] = psit_nR.data

        P_ani = self.calculate_projections(psit_nR)
        P2_ani = P_ani.new()
        self.dSinv(P_ani, out_ani=P2_ani)  # P2 is ΔC_ij @ P_nj
        self.pt_aiX.add_to(out_nR, P2_ani)


class CSCGAdapter:

    """ Adapter in order to reuse the CSCG solver from the old code

    """

    def __init__(self,
                 desc: UGDesc,
                 propagator: FDNumpyPropagator):
        self.solver = CSCG()
        self.solver.initialize(desc._gd, nulltimer)
        self.desc = desc
        self.propagator = propagator
        self._time_step = None

    def solve(self,
              init_guess_nR: UGArray,
              rhs_nR: UGArray,
              out_nR: UGArray,
              time_step: float):
        self._time_step = time_step
        out_nR.data[:] = init_guess_nR.data
        self.solver.solve(self, out_nR.data, rhs_nR.data)
        self._time_step = None

    def dot(self, psit_nR: np.ndarray, out_nR: np.ndarray):
        """Applies the propagator matrix to the given wavefunctions.

        (S + i H dt/2 ) psi

        Parameters
        ----------
        psi: List of coarse grids
            the known wavefunctions
        psin: List of coarse grids
            the result ( S + i H dt/2 ) psi

        """
        assert self._time_step is not None
        _psit_nR = self.desc.from_data(psit_nR)
        _out_nR = self.desc.from_data(out_nR)
        self.propagator.dot(_psit_nR, _out_nR, self._time_step)

    def apply_preconditioner(self, psi, psin):
        self.propagator.apply_preconditioner(psi, psin)


class ECNAlgorithm(TDAlgorithm):

    def propagate_wfs(self,
                      time_step: float,
                      state: DFTState,
                      pot_calc: PotentialCalculator,
                      wf_propagator: WaveFunctionPropagator):
        for wfs in state.ibzwfs:
            wf_propagator.propagate(wfs, time_step)


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
        symmetry = state.ibzwfs.ibz.symmetries.symmetry
        natoms = symmetry.a_sa.shape[1]
        symmetry.op_scc = np.eye(3)[None, ...]
        symmetry.ft_sc = np.zeros((1, 3))
        symmetry.a_sa = np.arange(natoms)[None, ...]
        state.ibzwfs.ibz.symmetries = Symmetries(symmetry)

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
            self.nkicks = 10
        elif isinstance(hamiltonian, FDHamiltonian):
            self.calculate_dipole_moment = self._calculate_dipole_moment
            self.mode = 'fd'
            self.nkicks = None
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

        state = dft.state
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
                                           dtype=complex)

        state = dft.state
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

            if self.mode == 'fd':
                self.nkicks = int(round(magnitude / 1.0e-4))
                if self.nkicks < 1:
                    self.nkicks = 1

            # Propagate kick
            return self.kick(cef)

    def kick(self,
             ext: ExternalPotential):
        """Kick with any external potential.

        Note that unless this function is called by absorption_kick, the kick
        is not logged in history

        Parameters
        ----------
        ext
            External potential
        """
        with self.timer('Kick'):
            self.log('----  Applying kick')
            self.log(f'----  {ext}')
            self.kick_ext = ext

            # For the kick, the propagator is always ECN
            td_algorithm = ECNAlgorithm()
            wf_propagator = self.kick_propagator(ext)

            # assert isinstance(self.pot_calc, FDPotentialCalculator)
            for l in range(self.nkicks):
                td_algorithm.propagate_wfs(1 / self.nkicks,
                                           state=self.state,
                                           pot_calc=self.pot_calc,
                                           wf_propagator=wf_propagator)
            td_algorithm.update_time_dependent_operators(self.state,
                                                         self.pot_calc)

            dipolemoment_xv = [
                self.calculate_dipole_moment(wfs)  # type: ignore
                for wfs in self.state.ibzwfs]
            dipolemoment_v = np.sum(dipolemoment_xv, axis=0)
            norm = np.sum(self.state.density.nct_aX.integral)
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
        kick_hamiltonian: type[Hamiltonian]
        if self.mode == 'lcao':
            kick_hamiltonian = LCAOKickHamiltonian(self.hamiltonian.basis,
                                                   self.state.ibzwfs,
                                                   ext,
                                                   self.pot_calc)
        elif self.mode == 'fd':
            kwargs = dict(kin_stencil=len(self.hamiltonian.kin.coef_p),
                          blocksize=self.hamiltonian.blocksize,
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
            norm = np.sum(self.state.density.nct_aX.integral)
            result = RTTDDFTResult(time=time,
                                   dipolemoment=dipolemoment_v,
                                   norm=norm)
            yield result

    def _calculate_dipole_moment(self, wfs: WaveFunctions) -> np.ndarray:
        dipolemoment_v = self.state.density.calculate_dipole_moment(
            self.pot_calc.fracpos_ac)

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
