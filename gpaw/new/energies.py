import numpy as np
from ase.units import Ha

# Contributions to free energy:
NAMES = {'kinetic', 'coulomb', 'zero', 'external', 'xc', 'entropy',
         'spinorbit'}

# Other allowed names:
OTHERS = {'band', 'kinetic_correction', 'extrapolate',
          'hybrid_kinetic_correction', 'hybrid_xc'}


class DFTEnergies:
    def __init__(self, **energies: float):
        self.energies = {}
        self.set(**energies)

    def set(self, **energies: float):
        assert energies.keys() <= NAMES | OTHERS, energies
        self.energies.update(energies)

    @property
    def total_free(self):
        energies = self.energies
        if 'kinetic' not in energies:
            # Use Kohn-Sham eq. to get kinetic energy as sum over
            # occupied eigenvalues + correction:
            energies['kinetic'] = (
                energies.get('band', np.nan) +
                energies.get('kinetic_correction', np.nan) +
                energies.get('hybrid_kinetic_correction', 0.0))
        if 'hybrid_xc' in energies:
            energies['xc'] += energies['hybrid_xc']
        return sum(energies.get(name, 0.0) for name in NAMES)

    def __repr__(self):
        return repr(self.energies)

    def summary(self, log) -> None:
        total_free = self.total_free
        total_extrapolated = total_free + energies.get('extrapolate', np.nan)
        for name in NAMES:
            e = energies.get(name)
            if name:
                log(f'{name + ":":10}   {e * Ha:14.6f}')
        log('----------------------------')
        log(f'Free energy: {total_free * Ha:14.6f}')
        log(f'Extrapolated:{total_extrapolated * Ha:14.6f}\n')
