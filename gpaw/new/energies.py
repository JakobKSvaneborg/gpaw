import numpy as np
from ase.units import Ha

# Contributions to free energy:
NAMES = {'kinetic', 'coulomb', 'zero', 'external', 'xc', 'entropy',
         'spinorbit'}

# Other allowerd names:
OTHERS = {'band', 'kinetic_correction',
          'hybrid_kinetic_correction', 'hybrid_xc'}


class DFTEnergies:
    def __init__(self, **energies: float):
        assert energies.keys() <= NAMES | OTHERS
        if 'kinetic' not in energies:
            # Use Kohn-Sham eq. to get kinetic energy as sum over
            # occupied eigenvalues + correction:
            energies['kinetic'] = (
                energies.get('band', np.nan) +
                energies.get('kinetic_correction', np.nan) +
                energies.get('hybrid_kinetic_correction', 0.0))
        if 'hybrid_xc' in energies:
            energies['xc'] += energies['hybrid_xc']
        self.total_free = sum(energies[name] for name in NAMES)
        self.total_extrapolated = (self.total_free +
                                   energies.get('extrapolate', np.nan))
        self.energies = energies

    def __repr__(self):
        return repr(self.energies)

    def summary(self, log):
        for name in NAMES:
            e = self.energies.get(name)
            if name:
                log(f'{name + ":":10}   {e * Ha:14.6f}')
        log('----------------------------')
        log(f'Free energy: {self.total_free * Ha:14.6f}')
        log(f'Extrapolated:{self.total_extrapolated * Ha:14.6f}\n')
