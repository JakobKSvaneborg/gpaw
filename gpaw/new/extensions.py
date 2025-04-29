class ExtensionParameter:
    def build(self, atoms):
        raise NotImplementedError

class Extension:
    name = 'unnamed extension'

    def __init__(self, atoms, domain_comm):
        ...

    def get_energy_contributions(self) -> dict[str, float]:
        raise NotImplementedError

    def force_contribution(self):
        raise NotImplementedError

    def move_atoms(self, atoms) -> None:
        raise NotImplementedError

"""
class D3Extension:
    def __init__(self, params, atoms):
        self.params = params
        self.atoms = atoms

    def update_forces_postscf(self, F_av):
        from ase.calculators.dftd3 import PureDFTD3
        atoms = self.atoms.copy()
        atoms.calc = PureDFTD3(xc=self.params.xc, **self.params.kwargs)
        F_av += self.atoms.get_forces()

    def update_energy_postscf(self, energies):
        from ase.calculators.dftd3 import PureDFTD3
        atoms = atoms.copy()
        atoms.calc = PureDFTD3(xc=self.params.xc, **self.params.kwargs)
        energies['D3'] += atoms.get_potential_energy()

@register
class D3(ExtensionParameter):
    def __init__(self, *, xc, **kwargs):
        self.xc = xc
        self.kwargs = kwargs

    def build(self, atoms):
        return D3Extension(self, atoms)
"""

