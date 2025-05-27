from ase import Atoms
from gpaw import GPAW, PW, FermiDirac

# cell and positions taken from c2db (c-axis reduced)
pos_av = [[0.0, 6.96017371e-04, 8.79377801e+00],
          [1.75031783e+00, 9.58254231e-01, 1.00912661e+01],
          [2.40358479e-16, 2.46817526e+00, 7.49524334e+00],
          [1.75031783e+00, 3.42574338e+00, 8.79271978e+00],
          [3.50063566e+00, 4.38341685e+00, 1.00921187e+01],
          [1.75031783e+00, 5.89595562e+00, 7.49437309e+00]]
cell = [[3.5006, 0.0, 0.0],
        [0.0, 6.8529, 0.0],
        [0.0, 0.0, 8.0]]
slab = Atoms('RhCl2RhCl2', positions=pos_av, cell=cell, pbc=[1, 1, 0])
slab.center(axis=2)
m = len(slab) * [0]
m[0] = 2
m[3] = 2
slab.set_initial_magnetic_moments(m)
calc = GPAW(mode=PW(600),
            xc='PBE',
            nbands=50,
            convergence={'bands': -10, 'density': 1.0e-6},
            occupations=FermiDirac(width=0.001),
            kpts={'size': (8, 4, 1), 'gamma': True},
            txt='gs_RhCl2.txt')
slab.calc = calc
slab.get_potential_energy()

calc.write('gs_RhCl2.gpw', mode='all')
