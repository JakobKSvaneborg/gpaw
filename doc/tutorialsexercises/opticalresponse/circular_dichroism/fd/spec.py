from gpaw.tddft.spectrum import rotatory_strength_spectrum

rotatory_strength_spectrum(['mm-x.dat', 'mm-y.dat',
                               'mm-z.dat'],
                               'rot_spec.dat',
                               folding='Gauss', width=0.2,
                               e_min=0.0, e_max=10.0, delta_e=0.01)
