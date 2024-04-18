from ase import Atoms
from ase.build import bulk
from gpaw import GPAW

atom = Atoms('Pt')
atom.center(vacuum=6.0)
atom.calc = GPAW(
    xc='PBE',
    mode={'name': 'pw', 'ecut': 600.0},
    nbands=-2,
    mixer={'backend': 'no-mixing'},
    occupations={'name': 'fixed-uniform'},
    hund=True,
    eigensolver={'name': 'etdm-fdpw', 'converge_unocc': True},
    symmetry='off',
    txt='pt-atom.txt')
e_atom = atom.get_potential_energy()
atom.calc.write('pt-atom.gpw')

bulk = bulk('Pt', 'fcc', a=3.92)
k = 8
bulk.calc = GPAW(
    xc='PBE',
    mode={'name': 'pw', 'ecut': 600.0},
    kpts=(k, k, k),
    txt='pt-bulk.txt')
e_bulk = bulk.get_potential_energy()

e_cohesive = e_atom - e_bulk
print(e_cohesive, 'eV')
