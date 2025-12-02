from ase import Atoms

from gpaw import GPAW

h = 0.2
n = 24
a = n * h
b = a / 2
H = Atoms('H', [(b - 0.1, b, b)], pbc=True, cell=(a, a, a))
calc = GPAW(mode='fd', nbands=1, gpts=(n, n, n), symmetry='off', txt='ltt.txt')
H.calc = calc
e0 = H.get_potential_energy()
for i in range(50):
    e = H.get_potential_energy()
    H.positions += (0.09123456789, 0.0423456789, 0.03456789)
assert abs(e - e0) < 0.0006
print(e, e0, e - e0)
