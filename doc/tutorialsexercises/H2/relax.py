# web-page: optimization.txt

from gpaw import restart
from ase.parallel import paropen as open
from ase.optimize import QuasiNewton


molecule, calc = restart('H2.gpw', txt='H2-relaxed.txt')

e2 = molecule.get_potential_energy()
d0 = molecule.get_distance(0, 1)

with open('optimization.txt', 'w') as fd:
    print('experimental bond length:', file=fd)
    print(f'hydrogen molecule energy: {e2:5.2f} eV', file=fd)
    print(f'bondlength              : {d0:5.2f} Ang', file=fd)

    # Find the theoretical bond length:
    relax = QuasiNewton(molecule, logfile='qn.log')
    relax.run(fmax=0.05)

    e2 = molecule.get_potential_energy()
    d0 = molecule.get_distance(0, 1)

    print(file=fd)
    print('PBE energy minimum:', file=fd)
    print(f'hydrogen molecule energy: {e2:5.2f} eV', file=fd)
    print(f'bondlength              : {d0:5.2f} Ang', file=fd)
