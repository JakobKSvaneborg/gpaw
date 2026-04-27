from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

from gpaw.external import ExternalPotential, create_external_potential
from gpaw.new.density import Density
from gpaw.new.energies import DFTEnergies
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.potential import Potential


RTTDDFTKickLike = Union['RTTDDFTKick', dict[str, Any]]


@dataclass()
class RTTDDFTState:
    """State of a Kohn-Sham system during RT-TDDFT."""
    ibzwfs: IBZWaveFunctions
    density: Density
    potential: Potential
    energies: DFTEnergies


@dataclass
class RTTDDFTKick:

    """ Class representing an RTTDDFT kick for logging purposes.
    """

    time: float  # Time of kick in atomic units
    potential: ExternalPotential
    gauge: str = 'length'

    def __post_init__(self):
        if self.gauge not in ['length', 'velocity']:
            raise ValueError('Only length and velocity gauge supported')
        if isinstance(self.potential, dict):
            self.potential = create_external_potential(**self.potential)

    def todict(self):
        return {'time': self.time,
                'gauge': self.gauge,
                'potential': self.potential.todict()}
