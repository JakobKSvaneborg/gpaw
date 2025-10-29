from __future__ import annotations

import warnings

from functools import cached_property
from types import SimpleNamespace
from typing import Any

import numpy as np

from gpaw.external import ConstantElectricField, create_absorption_kick
from gpaw.new.backwards_compatibility import FakePoisson
from gpaw.mpi import world
from gpaw.new.ase_interface import ASECalculator
from gpaw.new.rttddft.rttddft import RTTDDFT
from gpaw.new.rttddft.td_algorithm import TDAlgorithmLike
from gpaw.tddft.units import as_to_au, autime_to_asetime
from gpaw.typing import Vector


class FakeTDHamiltonian:

    def __init__(self,
                 rttddft: RTTDDFT):
        self._rttddft = rttddft
        self.poisson = FakePoisson()

    def get_hamiltonian_matrix(self, kpt, time, addfxc=True, addpot=True,
                               scale=True):
        hamiltonian = self._rttddft.hamiltonian
        ham_calc = hamiltonian.create_hamiltonian_matrix_calculator(
            self._rttddft.state.potential)
        wfs = self._rttddft.state.ibzwfs.wfs_qs[kpt.q][kpt.s]
        H_MM = ham_calc.calculate_matrix(wfs)
        return H_MM.data


class RTTDDFTAdapter:
    """ Adapter to use old-GPAW code with new RTTDDFT """

    def __init__(self,
                 rttddft: RTTDDFT):
        self._rttddft = rttddft
        self.td_hamiltonian = FakeTDHamiltonian(rttddft)
        self.observers: list[Any] = []
        self.action = ''
        if world.size > 1:
            raise NotImplementedError
        self.tddft_initialized = False

        msg = ('Using compabilitity wrapper for RTTDDFT. Some options will '
               'be ignored. The recommended way of using the new RTTDDFT '
               'interface outside of tests is via gpaw.new.rttddft.RTTDDFT.')
        warnings.warn(msg)

    @property
    def world(self):
        return world

    @cached_property
    def wfs(self):
        from gpaw.new.backwards_compatibility import FakeWFS
        state = self._rttddft.state
        wfs = FakeWFS(state.ibzwfs,
                      state.density,
                      state.potential,
                      self._rttddft.pot_calc.setups,
                      world,
                      SimpleNamespace(occ=SimpleNamespace()),
                      self._rttddft.hamiltonian,
                      self.atoms)
        return wfs

    @property
    def density(self):
        from gpaw.new.backwards_compatibility import FakeDensity
        state = self._rttddft.state
        return FakeDensity(ibzwfs=state.ibzwfs,
                           density=state.density,
                           potential=state.potential,
                           pot_calc=self._rttddft.pot_calc)

    @property
    def hamiltonian(self):
        from gpaw.new.backwards_compatibility import FakeHamiltonian
        state = self._rttddft.state
        energies = dict(self._rttddft.state.energies._energies)
        energies['kinetic0'] = energies.pop('kinetic_correction')
        energies = {f'e_{key}': value
                    for key, value in energies.items()
                    if key not in ['spinorbit']}
        return FakeHamiltonian(state.ibzwfs, state.density,
                               state.potential, self._rttddft.pot_calc,
                               **energies)

    def attach(self, function, n=1, *args, **kwargs):
        """Register observer function to run during the propagation.

        Call *function* using *args* and
        *kwargs* as arguments.

        If *n* is positive, then
        *function* will be called every *n* SCF iterations + the
        final iteration if it would not be otherwise

        If *n* is negative, then *function* will only be
        called on iteration *abs(n)*.

        If *n* is 0, then *function* will only be called
        on convergence"""

        try:
            slf = function.__self__
        except AttributeError:
            pass
        else:
            if slf is self:
                # function is a bound method of self.  Store the name
                # of the method and avoid circular reference:
                function = function.__func__.__name__

        # Replace self in args with another unique reference
        # to avoid circular reference
        if not hasattr(self, 'self_ref'):
            self.self_ref = object()
        self_ = self.self_ref
        args = tuple([self_ if arg is self else arg for arg in args])

        self.observers.append((function, n, args, kwargs))

    def call_observers(self, iter, final=False):
        """Call all registered callback functions."""
        for function, n, args, kwargs in self.observers:
            call = False
            # Call every n iterations, including the last
            if n > 0:
                if ((iter % n) == 0) != final:
                    call = True
            # Call only on iteration n
            elif n < 0 and not final:
                if iter == abs(n):
                    call = True
            # Call only on convergence
            elif n == 0 and final:
                call = True
            if call:
                if isinstance(function, str):
                    function = getattr(self, function)
                # Replace self reference with self
                self_ = self.self_ref
                args = tuple([self if arg is self_ else arg for arg in args])
                function(*args, **kwargs)

    def tddft_init(self):
        if self.tddft_initialized:
            return

        # In principle we should update the operators here
        # to be consistent with the old code
        self._rttddft.td_algorithm.update_time_dependent_operators(
            self._rttddft.state, self._rttddft.pot_calc)

        self.action = 'init'
        self.call_observers(self.niter)

        self.tddft_initialized = True

    def absorption_kick(self, kick_strength: Vector):
        """Kick with a weak electric field.

        Parameters
        ----------
        kick_strength
            Strength of the kick in atomic units
        """
        self.tddft_init()
        # TODO LCAOTDDFT does niter += for absorption_kick, but TDDFT does not

        # Kick and store history

        kw = dict()
        if self.mode == 'fd':
            # Different behavior between LCAO and FD. See #1423
            kw['nkicks'] = int(round(np.linalg.norm(kick_strength) / 1.0e-4))
            if kw['nkicks'] < 1:
                kw['nkicks'] = 1

        # Propagate kick
        potential = create_absorption_kick(kick_strength)
        result = self._rttddft.kick(potential, **kw)
        print(result)

        # Store kick in history
        self._rttddft.history.register_kick(potential)

        # Call observers after kick
        self.action = 'kick'
        self.call_observers(self.niter)

    def propagate(self, time_step: float = 10.0, iterations: int = 2000):
        """Propagate the electronic system.

        Parameters
        ----------
        time_step
            Time step in attoseconds
        iterations
            Number of propagation steps
        """
        self.tddft_init()

        time_step = time_step * as_to_au * autime_to_asetime

        print(f'---- Starting propagation {iterations} steps of '
              f'{time_step:.5f}Å√(u/eV)')
        print(f'---- Using {self._rttddft.td_algorithm}')
        for result in self._rttddft.ipropagate(time_step, iterations):
            print(result)

            # Call registered callback functions
            self.action = 'propagate'
            self.call_observers(self.niter)

    def __getattr__(self, attr):
        if attr in ['niter', 'time']:
            return getattr(self._rttddft.history, attr)
        if attr in ['kick_strength', 'kick_gauge']:
            try:
                # Return last kick
                kick = self._rttddft.history.kicks[-1]
                if attr == 'kick_gauge':
                    return kick.gauge
                assert isinstance(kick.potential, ConstantElectricField)
                # Magnitude in atomic units
                magnitude = kick.potential.strength
                # Normalized direction
                direction_v = kick.potential.direction_v
                return magnitude * direction_v
            except IndexError:
                # There have been no kicks
                return None
        elif attr in ['setups']:
            return getattr(self._rttddft.pot_calc, attr)
        else:
            return getattr(self._rttddft, attr)

    def write(self,
              filename: str,
              mode=None):
        # Ignore mode option
        return self._rttddft.write(filename)

    @classmethod
    def from_dft_calculation(cls,
                             calc: ASECalculator,
                             propagator: TDAlgorithmLike = None):
        rttddft = RTTDDFT.from_dft_calculation(calc,
                                               td_algorithm=propagator)
        return cls(rttddft)

    @classmethod
    def from_dft_file(cls,
                      filepath: str,
                      propagator: TDAlgorithmLike = None):
        rttddft = RTTDDFT.from_dft_file(filepath,
                                        td_algorithm=propagator)
        return cls(rttddft)

    @classmethod
    def from_rttddft_file(cls,
                          filepath: str):
        rttddft = RTTDDFT.from_rttddft_file(filepath)
        return cls(rttddft)

    @classmethod
    def from_file(cls,
                  filepath: str,
                  **kwargs):
        if 'propagator' in kwargs:
            kwargs['td_algorithm'] = kwargs.pop('propagator')
        rttddft = RTTDDFT.from_file(filepath, **kwargs)
        return cls(rttddft)
