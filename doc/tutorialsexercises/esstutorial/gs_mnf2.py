from ase.spacegroup import crystal
# from ase.visualize import view
from gpaw import GPAW, PW

# Structure taken from J. Applied Crys. 21 (1988) 975-979
Mn_c = [0, 0, 0]  # Wyckoff positions in relative coordinates
F_c = [0.30443, 0.30443, 0]
a, c = 4.8736, 3.2998  # Cell parameters in Å

mnf2 = crystal('MnF2',
               cellpar=[a, a, c, 90, 90, 90],
               basis=[Mn_c, F_c],
               primitive_cell=True,
               spacegroup=136,
               pbc=True)

# Initial magnetic moments in Bohr magnetons
magmoms = [5, -5, 0, 0, 0, 0]
mnf2.set_initial_magnetic_moments(magmoms)

# view(mnf2) # View structure with ASE's build-in GUI

# Calculator
calc = GPAW(mode=PW(400),
            xc='LDA',
            setups={'Mn': ':d,6.0'},
            nbands=80,
            kpts={'size': (4, 4, 4), 'gamma': True},
            txt='gs_afm.txt',
)

# Attach calculator and do the calculation
mnf2.calc = calc
mnf2.get_potential_energy()

# Save everything (also wavefunction) to .gpw file
calc.write('gs_afm.gpw', mode='all')
