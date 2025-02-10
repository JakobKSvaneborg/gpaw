from __future__ import annotations

from gpaw.new.density import Density
from gpaw.new.hamiltonian import Hamiltonian
from gpaw.new.potential import Potential
from gpaw.new.energies import DFTEnergies
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.pot_calc import PotentialCalculator


class Eigensolver:
    direct = False

    def iterate(self,
                ibzwfs: IBZWaveFunctions,
                density: Density,
                potential: Potential,
                hamiltonian: Hamiltonian,
                pot_calc: PotentialCalculator,
                energies: DFTEnergies) -> tuple[float, DFTEnergies]:
        raise NotImplementedError

    def postprocess(self, ibzwfs, density, potential, hamiltonian):
        pass
