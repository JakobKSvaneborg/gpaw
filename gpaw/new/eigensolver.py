from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from gpaw.new.calculation import DFTState


class Eigensolver:
    direct = False

    def iterate(self, state: DFTState, hamiltonian) -> float:
        raise NotImplementedError

    def initialize_etdm(self, *args, **kwargs):
        pass

    def postprocess(self, state, hamiltonian):
        pass
