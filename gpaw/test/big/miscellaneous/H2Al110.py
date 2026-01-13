from ase import Atoms

from gpaw import GPAW

a = 4.00
d = a / 2**0.5
z = 1.1
b = 1.5

slab = Atoms('Al10H2',
             [(0, 0, 0),
              (a, 0, 0),
              (a / 2, d / 2, -d / 2),
              (3 * a / 2, d / 2, -d / 2),
              (0, 0, -d),
              (a, 0, -d),
              (a / 2, d / 2, -3 * d / 2),
              (3 * a / 2, d / 2, -3 * d / 2),
              (0, 0, -2 * d),
              (a, 0, -2 * d),
              (a / 2 - b / 2, 0, z),
              (a / 2 + b / 2, 0, z)],
             cell=(2 * a, d, 5 * d),
             pbc=(1, 1, 1))
calc = GPAW(mode='fd',
            h=0.25,
            nbands=28,
            kpts=(2, 6, 1))
slab.calc = calc
e = slab.get_potential_energy()
niter = calc.get_number_of_iterations()
assert len(calc.get_k_point_weights()) == 3

for i in range(1):
    slab.positions[-2, 0] -= 0.01
    slab.positions[-1, 0] += 0.01
    e = slab.get_potential_energy()

print(e, niter)
assert abs(e - -44.694) < 0.0015
