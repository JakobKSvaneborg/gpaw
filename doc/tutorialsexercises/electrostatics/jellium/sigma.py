from ase.io import read
e0 = read('bulk.txt').get_potential_energy()
e = read('surface.txt').get_potential_energy()
a = 1.6
L = 10 * a
sigma = (e - L / a * e0) / 2 / a**2
print(f'{1000 * sigma:.2f} meV/Ang^2')
print(f'{sigma / 6.24150974e-05:.1f} erg/cm^2')
assert abs(sigma - 0.0054) < 0.0001
