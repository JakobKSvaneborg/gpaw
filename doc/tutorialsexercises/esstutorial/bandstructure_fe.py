from gpaw import GPAW

# Restart from ground state and fix the KS Hamiltonian
calc = GPAW('gs_fe.gpw').fixed_density(
    symmetry='off',
    kpts={'path': 'GHNG', 'npoints': 60})

# Write the results to bs.gpw and plot bands
bs = calc.band_structure()
calc.write('bs_fe.gpw')
bs.plot(filename='bandstructure_fe.png', show=True, emin=0.0, emax=14.0)
