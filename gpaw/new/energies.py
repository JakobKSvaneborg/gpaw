import numpy as np
from ase.units import Ha


class DFTEnergies:
    def __init__(self, **energies: float):
        if 'kinetic' not in energies:
            energies['kinetic'] = (energies.pop('band', np.nan) +
                                   energies.pop('kinetic_correction', np.nan))
        self.total_free = sum(energies.values())
        self.total_extrapolated = (self.total_free +
                                   energies.get('extrapolation', np.nan))
        self.energies = energies

    def summary(self, log):
        for name, e in self.energies.items():
            self.log(f'{name + ":":10}   {e * Ha:14.6f}')
        self.log('----------------------------')
        self.log(f'Free energy: {self.total_free * Ha:14.6f}')
        self.log(f'Extrapolated:{self.total_extrapolated * Ha:14.6f}\n')
