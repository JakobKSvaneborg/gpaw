from __future__ import annotations


class Eigensolver:
    direct = False

    def iterate(self,
                ibzwfs,
                density,
                potential,
                hamiltonian) -> float:
        raise NotImplementedError

    def initialize_etdm(self, *args, **kwargs):
        pass

    def postprocess(self, ibzwfs, density, potential, hamiltonian):
        pass
