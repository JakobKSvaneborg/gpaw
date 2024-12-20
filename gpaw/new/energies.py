"""PAW-DFT energy contributions."""

import numpy as np
from ase.units import Ha

# Contributions to free energy:
NAMES = {'kinetic', 'coulomb', 'zero', 'external', 'xc', 'entropy',
         'spinorbit'}

# Other allowed names:
OTHERS = {'band', 'kinetic_correction', 'extrapolation',
          'hybrid_kinetic_correction', 'hybrid_xc'}


class DFTEnergies:
    def __init__(self, **energies: float):
        self._energies: dict[str, float] = {}
        self._total_free: float | None
        self.set(**energies)

    def set(self, **energies: float) -> None:
        assert energies.keys() <= NAMES | OTHERS, energies
        self._energies.update(energies)
        self._total_free = None

    @property
    def total_free(self) -> float:
        if self._total_free is None:
            energies = self._energies.copy()
            if 'kinetic' not in energies:
                # Use Kohn-Sham eq. to get kinetic energy as sum over
                # occupied eigenvalues + correction:
                energies['kinetic'] = (
                    energies['band'] +
                    energies['kinetic_correction'] +
                    energies.get('hybrid_kinetic_correction', 0.0))
            if 'hybrid_xc' in energies:
                energies['xc'] += energies['hybrid_xc']
            self._total_free = sum(energies.get(name, 0.0) for name in NAMES)
        return self._total_free

    @property
    def total_extrapolated(self) -> float:
        return self.total_free + self._energies['extrapolation']

    def __repr__(self) -> str:
        s = ', '.join(f'{k}={v}' for k, v in self._energies.items())
        return f'DFTEnergies({s})'

    def summary(self, log) -> None:
        for name in NAMES:
            e = self._energies.get(name)
            if e is not None:
                log(f'{name + ":":10}   {e * Ha:14.6f}')
        log('----------------------------')
        log(f'Free energy: {self.total_free * Ha:14.6f}')
        log(f'Extrapolated:{self.total_extrapolated * Ha:14.6f}\n')

    def write_to_gpw(self, writer):
        writer.write(**{name: e * Ha for name, e in self._energies.items()})
