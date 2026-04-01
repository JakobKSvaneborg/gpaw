from gpaw import GPAW, PW, MethfesselPaxton
from ase.spacegroup import crystal
from ase.io import write

a = 4.55643
mnsi = crystal(['Mn', 'Si'],
               [(0.1380, 0.1380, 0.1380), (0.84620, 0.84620, 0.84620)],
               spacegroup=198,
               cellpar=[a, a, a, 90, 90, 90])

mnsi.set_initial_magnetic_moments([1.0, ] * len(mnsi))

mnsi.calc = GPAW(xc='PBE',
                 kpts=(2, 2, 2),
                 mode=PW(800),
                 occupations=MethfesselPaxton(width=0.005),
                 txt='mnsi.txt')

mnsi.get_potential_energy()
mnsi.calc.write('mnsi.gpw')
v = mnsi.calc.get_electrostatic_potential()
write('mnsi.cube', mnsi, data=v)

assert abs(v.max() - 13.43) < 0.01
