from gpaw.new.input_parameters import register
from ase.units import Hartree, Bohr


class Extension:
    name = 'unnamed extension'

    def get_energy_contributions(self) -> dict[str, float]:
        raise NotImplementedError

    def force_contribution(self):
        raise NotImplementedError

    def move_atoms(self, relpos_ac) -> None:
        raise NotImplementedError


class ExtensionParameter:
    def build(self, atoms, domain_comm) -> Extension:
        raise NotImplementedError


@register
class D3(ExtensionParameter):
    def __init__(self, *, xc, **kwargs):
        self.xc = xc
        self.kwargs = kwargs

    def todict(self) -> dict:
        return {'xc': self.xc, **self.kwargs}

    def build(self, atoms, domain_comm):
        atoms = atoms.copy()
        class D3Extension(Extension):
            def __init__(self):
                super().__init__()
                self._calculate(atoms)

            def _calculate(_self, atoms):
                from ase.calculators.dftd3 import PureDFTD3
                atoms.calc = PureDFTD3(xc=self.xc, **self.kwargs)
                # XXX params.xc should be taken directly from the calculator.
                # XXX What if this is changed via set?
                _self.F_av = atoms.get_forces() / Hartree * Bohr
                _self.E = atoms.get_potential_energy() / Hartree

            def get_energy_contributions(_self) -> dict[str, float]:
                return {f'D3 (xc={self.xc})': _self.E}

            def force_contribution(self):
                if domain_comm.rank == 0:
                    return self.F_av
                else:
                    return np.zeros_like(self.F_av)

            def move_atoms(self, relpos_ac) -> None:
                atoms.set_scaled_positions(relpos_ac)
                self._calculate(atoms)

        return D3Extension()
