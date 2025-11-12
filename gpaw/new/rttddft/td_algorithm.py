from __future__ import annotations

from abc import ABC
from typing import Any, Union

from gpaw.new.hamiltonian import Hamiltonian
from gpaw.new.pot_calc import PotentialCalculator
from gpaw.new.rttddft.dataclasses import RTTDDFTState
from gpaw.new.rttddft.wf_propagator import build_wf_propagator


TDAlgorithmLike = Union[None, str, 'TDAlgorithm', dict[str, Any]]


def create_td_algorithm(name: TDAlgorithmLike,
                        **kwargs) -> TDAlgorithm:
    if name is None:
        return create_td_algorithm('sicn')
    elif isinstance(name, TDAlgorithm):
        return name
    elif isinstance(name, dict):
        kwargs.update(name)
        return create_td_algorithm(**kwargs)

    name = name.lower()
    if name == 'sicn':
        return SICNAlgorithm(**kwargs)
    elif name == 'ecn':
        return ECNAlgorithm(**kwargs)
    else:
        raise ValueError(f'Unknown propagation algorithm: {name}')


class TDAlgorithm(ABC):

    """ Propagation algorithm for the state

    Parameters
    ----------
    implementation
        Name of wave function propagator implementation
    """

    def __init__(self,
                 implementation: str = 'numpy'):
        self.implementation = implementation

    def propagate(self,
                  time_step: float,
                  state: RTTDDFTState,
                  pot_calc: PotentialCalculator,
                  hamiltonian: Hamiltonian):
        r""" Perform one propagation step, i.e.


        (1) Calculate propagator :math:`U[H(t)]`
        (2) Update wavefunctions :math:`ψ_n(t+\Delta t) = U[H(t)] ψ_n(t)`
        (3) Update density and hamiltonian :math:`H(t+dt)`

        .. math::

            U(0^+, 0) = \hat{T} \mathrm{exp}\left[ -i S^{-1}
            \int_t^{t+\Delta t} \mathrm{d}\tau \hat{H}(\tau)
            \right]
        """
        raise NotImplementedError

    def update_time_dependent_operators(self,
                                        state: RTTDDFTState,
                                        pot_calc: PotentialCalculator):
        # Update density
        state.density.update(state.ibzwfs)

        # Calculate Hamiltonian H(t+dt) = H[n[Phi_n]]
        state.potential, state.energies, _ = pot_calc.calculate(
            state.density, state.ibzwfs, vHt_x=state.potential.vHt_x)

    def propagate_wfs(self,
                      time_step: float,
                      state: RTTDDFTState,
                      hamiltonian: Hamiltonian):
        wf_propagator = build_wf_propagator(self.implementation,
                                            hamiltonian, state)
        for wfs in state.ibzwfs:
            wf_propagator.propagate(wfs, wfs, time_step)

    def get_description(self):
        return self.__class__.__name__

    def __str__(self) -> str:
        return (f'{self.get_description()} '
                f'({self.implementation} implementation)')

    def todict(self):
        raise NotImplementedError


class ECNAlgorithm(TDAlgorithm):

    """ Explicit Crank-Nicolson algorithm

    Crank-Nicolson propagator, which approximates the time-dependent
    Hamiltonian to be unchanged during one iteration step.

    Parameters
    ----------
    implementation
        Name of wave function propagator implementation
    """

    def propagate(self,
                  time_step: float,
                  state: RTTDDFTState,
                  pot_calc: PotentialCalculator,
                  hamiltonian: Hamiltonian):
        # Propagate wave functions one timestep; ψ(t) -> ψ(t + dt)
        self.propagate_wfs(time_step, state, hamiltonian)

        # Calculate density and Hamiltonian at t + dt
        self.update_time_dependent_operators(state, pot_calc)

    def todict(self):
        return {'name': 'ecn',
                'implementation': self.implementation}


class SICNAlgorithm(TDAlgorithm):

    """Semi-implicit Crank-Nicolson propagator

    Crank-Nicolson propagator, which first approximates the time-dependent
    Hamiltonian to be unchanged during one iteration step to predict future
    wavefunctions. Then the approximations for the future wavefunctions are
    used to approximate the Hamiltonian at the middle of the time step.

    Parameters
    ----------
    implementation
        Name of wave function propagator implementation
    """

    def propagate(self,
                  time_step: float,
                  state: RTTDDFTState,
                  pot_calc: PotentialCalculator,
                  hamiltonian: Hamiltonian):
        # Copy wave functions
        prev_wfs_list = [wfs.copy() for wfs in state.ibzwfs]

        # Copy Hamiltonian
        prev_potential = state.potential.copy()

        # Propagate wave functions one timestep; ψ(t) -> ψ(t + dt)
        self.propagate_wfs(time_step, state, hamiltonian)

        # Calculate density and Hamiltonian at t + dt
        self.update_time_dependent_operators(state, pot_calc)

        # Average Hamiltonian at t and t + dt
        state.potential.vt_sR.data[:] += prev_potential.vt_sR.data
        state.potential.vt_sR.data[:] *= 0.5
        state.potential.dH_asii.data[:] += prev_potential.dH_asii.data
        state.potential.dH_asii.data[:] *= 0.5

        # Restore the previous wave functions
        state.ibzwfs._wfs_u = prev_wfs_list

        # Propagate wave functions one timestep; ψ(t) -> ψ(t + dt)
        # using the averaged Hamiltonian
        self.propagate_wfs(time_step, state, hamiltonian)

        # Calculate density and Hamiltonian at t + dt
        self.update_time_dependent_operators(state, pot_calc)

    def todict(self):
        return {'name': 'sicn',
                'implementation': self.implementation}
