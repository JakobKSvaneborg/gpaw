from __future__ import annotations

import numpy as np

from gpaw.typing import Vector


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
        kick_strength = self.kick_strength
        if kick_strength is not None:
            kick_strength = kick_strength.tolist()
        return {'niter': self.niter, 'time': self.time,
                'kick_strength': kick_strength}
