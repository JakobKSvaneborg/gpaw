from __future__ import annotations

from gpaw.external import ExternalPotential
from gpaw.new.rttddft.dataclasses import RTTDDFTKick, RTTDDFTKickLike


class RTTDDFTHistory:

    """ Representation of the history of a RT-TDDFT calculation.
    The class stores the curent time and the number of propagation steps,
    as well as a list of previous kicks.
    """

    def __init__(self) -> None:
        self._kicks: list[RTTDDFTKick] = []
        self._niter: int = 0
        self._time: float = 0.0

    @property
    def time(self) -> float:
        """ Current simulation time in atomic units. """
        return self._time

    @property
    def niter(self) -> int:
        """ Number of propagation steps. """
        return self._niter

    @property
    def kicks(self) -> list[RTTDDFTKick]:
        """ Kicks that have been done. """
        return self._kicks

    def register_kick(self,
                      potential: ExternalPotential,
                      gauge: str = 'length'):
        """ Store the kick in history.

        Parameters
        ----------
        potential
            External potential.
        gauge
            Kick gauge.
        """
        kick = RTTDDFTKick(self.time, potential=potential, gauge=gauge)
        self._kicks.append(kick)

    def propagate(self,
                  time_step: float) -> float:
        """ Increment the number of propagation steps and simulation time
        in history.

        Parameters
        ----------
        time_step
            Time step in atomic units.

        Returns
        -------
        float
            The new simulation time in atomic units.
        """
        self._niter += 1
        self._time += time_step

        return self.time

    def todict(self):
        kicks = [kick.todict() for kick in self.kicks]
        return {'niter': self.niter, 'time': self.time, 'kicks': kicks}

    @classmethod
    def from_values(cls,
                    kicks: list[RTTDDFTKickLike],
                    niter: int,
                    time: float) -> RTTDDFTHistory:
        history = cls()
        history._kicks += [kick if isinstance(kick, RTTDDFTKick)
                           else RTTDDFTKick(**kick) for kick in kicks]
        history._niter = niter
        history._time = time
        return history
