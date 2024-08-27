from ase.parallel import paropen
from gpaw import GPAW

layer = GPAW('Fe_gs.gpw', txt=None).atoms
path = layer.cell.bandpath(['G', 'H', (1, 0, 0)], npoints=1000)
kpts = path.kpts
(x, X, labels) = path.get_linear_kpoint_axis()

calc = GPAW('Fe_gs.gpw').fixed_density(
    kpts=kpts,
    symmetry='off',
    txt='Fe_bands.txt',
    parallel={'band': 1})

calc.write('Fe_bands.gpw')

f = paropen('Fe_kpath.dat', 'w')
for k in x:
    print(k, file=f)
f.close()

f = paropen('Fe_highsym.dat', 'w')
for k in X:
    print(k, file=f)
f.close()
