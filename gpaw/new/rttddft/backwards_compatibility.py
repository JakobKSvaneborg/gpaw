from __future__ import annotations

from typing import Any
import numpy as np

from gpaw.mpi import world
from gpaw.new.ase_interface import ASECalculator
from gpaw.new.rttddft.rttddft import RTTDDFT
from gpaw.new.rttddft.td_algorithm import TDAlgorithmLike
from gpaw.tddft.units import as_to_au, autime_to_asetime
from gpaw.typing import Vector


class PoissonAdapter:

    def get_description(self):
        return ''


class HamiltonianAdapter:

    def __init__(self,
                 rttddft: RTTDDFT):
        self.poisson = PoissonAdapter()


class DensityAdapter:

    def __init__(self,
                 rttddft: RTTDDFT):
        self._density = rttddft.state.density
        self._pot_calc = rttddft.pot_calc

    @property
    def density(self):
        return self._density

    @property
    def pot_calc(self):
        return self._pot_calc

    def __getattr__(self, attr):
        if attr == 'finegd':
            return self.pot_calc.fine_grid._gd
        if attr == 'nt_sg':
            # Intepolate density
            nt_sr = self.pot_calc.interpolate(self.nt_sR)

            # Compute pseudo charge
            pseudo_charge = nt_sr.integrate().sum()
            ccc_aL = self.density.calculate_compensation_charge_coefficients()
            comp_charge = (4 * np.pi)**0.5 * sum(float(ccc_L[0])
                                                 for ccc_L in ccc_aL.values())
            comp_charge = ccc_aL.layout.atomdist.comm.sum_scalar(comp_charge)

            # Normalize
            nt_sr.data *= -comp_charge / pseudo_charge
            return nt_sr.data
        if attr == 'rhot_g':
            rhot_g = self.pot_calc.fine_grid.empty()
            rhot_g.data[:] = self.nt_sg.sum(axis=0)
            ccc_aL = self.density.calculate_compensation_charge_coefficients()
            self.pot_calc.ghat_aLr.add_to(rhot_g, ccc_aL)
            return rhot_g.data

        return getattr(self._density, attr)


class RTTDDFTAdapter:
    """ Adapter to use old-GPAW code with new RTTDDFT """

    def __init__(self,
                 rttddft: RTTDDFT):
        self._rttddft = rttddft
        self._density = DensityAdapter(rttddft)
        self._hamiltonian = HamiltonianAdapter(rttddft)
        self.observers: list[Any] = []
        self.action = ''
        if world.size > 1:
            raise NotImplementedError
        self.tddft_initialized = False

    @property
    def world(self):
        return world

    @property
    def density(self):
        return self._density

    @property
    def hamiltonian(self):
        return self._hamiltonian

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
        result = self._rttddft.absorption_kick(kick_strength)
        print(result)

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
        if attr in ['niter', 'time', 'kick_strength']:
            return getattr(self._rttddft.history, attr)
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
