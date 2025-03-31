from gpaw.tddft.spectrum import rotatory_strength_spectrum

for gauge in ['length', 'velocity']:
    rotatory_strength_spectrum(['mm-{gauge}_x.dat', 'mm-{gauge}_y.dat', 'mm-{gauge}_z.dat'],
                               'rot_spec-{gauge}.dat',
                               folding='Gauss', width=0.2,
                               e_min=0.0, e_max=10.0, delta_e=0.01)
