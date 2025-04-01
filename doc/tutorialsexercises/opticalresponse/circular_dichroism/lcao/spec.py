from gpaw.tddft.spectrum import rotatory_strength_spectrum

for gauge in ['length', 'velocity']:
    rotatory_strength_spectrum([f'mm-{gauge}_x.dat', f'mm-{gauge}_y.dat',
                               f'mm-{gauge}_z.dat'],
                               f'rot_spec-{gauge}.dat',
                               folding='Gauss', width=0.2,
                               e_min=0.0, e_max=10.0, delta_e=0.01)
