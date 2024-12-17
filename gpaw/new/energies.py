from gpaw.mpi import broadcast_float, world


class EnergyContributions:
    def __init__(self, contributions):
        self.contributions = contributions
pseudo_energies = pseudo_energies
        self.corrections = corrections
        self.stress = np.nan

    @cached_property
    def total_free(self):
        self._update()
        return
def combine_energies(potential: Potential,
                     ibzwfs: IBZWaveFunctions) -> dict[str, float]:
    """Add up energy contributions."""
    energies = potential.energies.copy()
    energies.pop('stress', 0.0)
    energies['kinetic'] += ibzwfs.energies['band']
    energies['kinetic'] += ibzwfs.energies.get('exx_kinetic', 0.0)
    energies['xc'] += (ibzwfs.energies.get('exx_vv', 0.0) +
                       ibzwfs.energies.get('exx_vc', 0.0) +
                       ibzwfs.energies.get('exx_cc', 0.0))
    energies['entropy'] = ibzwfs.energies['entropy']
    energies['total_free'] = sum(energies.values())
    energies['total_extrapolated'] = (energies['total_free'] +
                                      ibzwfs.energies['extrapolation'])
    return energies
        energies = combine_energies(self.potential, self.ibzwfs)

        self.log('Energy contributions relative to reference atoms:',
                 f'(reference = {self.setups.Eref * Ha:.6f})\n')

        for name, e in energies.items():
            if not name.startswith('total') and name != 'stress':
                self.log(f'{name + ":":10}   {e * Ha:14.6f}')
        total_free = energies['total_free']
        total_extrapolated = energies['total_extrapolated']
        self.log('----------------------------')
        self.log(f'Free energy: {total_free * Ha:14.6f}')
        self.log(f'Extrapolated:{total_extrapolated * Ha:14.6f}\n')

        total_free = broadcast_float(total_free, self.comm)
        total_extrapolated = broadcast_float(total_extrapolated, self.comm)

