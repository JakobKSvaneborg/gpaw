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

    def build(self, atoms, domain_comm):
        atoms = atoms.copy()
        class D3Extension(Extension):
            def __init__(self):
                super().__init__()
                self._calculate(atoms)

            def _calculate(self, atoms):
                from ase.calculators.dftd3 import PureDFTD3
                atoms.calc = PureDFTD3(xc=self.params.xc, **self.params.kwargs)
                # XXX params.xc should be taken directly from the calculator.
                # XXX What if this is changed via set?
                self.F_av = atoms.get_forces()
                self.E = atoms.get_potential_energy()

            def get_energy_contributions(_self) -> dict[str, float]:
                return {f'D3 (xc={self.xc})': self.E}

            def force_contribution(self):
                if domain_comm.rank == 0:
                    return self.F_av
                else:
                    return np.zeros_like(self.F_av)

            def move_atoms(self, relpos_ac) -> None:
                atoms.set_scaled_positions(relpos_ac)

        return D3Extension()
