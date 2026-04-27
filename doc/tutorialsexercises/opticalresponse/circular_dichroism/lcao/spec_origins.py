from gpaw.tddft.spectrum import rotatory_strength_spectrum

for gauge in ['length', 'velocity']:
    for tag in ['COM', 'COM+x', 'COM+y', 'COM+z', '123']:
        rotatory_strength_spectrum([f'mm-{tag}-{gauge}_{k}.dat'
                                   for k in 'xyz'],
                                   f'rot_spec-{tag}_{gauge}.dat',
                                   folding='Gauss', width=0.2,
                                   e_min=0.0, e_max=10.0, delta_e=0.01)
