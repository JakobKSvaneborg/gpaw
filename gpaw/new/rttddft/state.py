from __future__ import annotations

from dataclasses import dataclass

from gpaw.new.density import Density
from gpaw.new.energies import DFTEnergies
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.potential import Potential


@dataclass()
class RTTDDFTState:
    """State of a Kohn-Sham system during RT-TDDFT."""
    ibzwfs: IBZWaveFunctions
    density: Density
    potential: Potential
    energies: DFTEnergies
