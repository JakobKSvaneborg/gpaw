from ase.build import molecule
from ase.parallel import paropen

from gpaw import GPAW, setup_paths, FermiDirac
from gpaw.utilities.adjust_cell import adjust_cell


setup_paths.insert(0, '.')

box = 5
h = 0.2
xc = 'PBE'
q = -1

atoms = molecule('SH2')
adjust_cell(atoms, box, h)

calc_gs = GPAW(mode='fd',
               nbands=-30,
               h=h,
               txt='h2s_gs.txt',
               xc=xc)

atoms.calc = calc_gs
e_gs = atoms.get_potential_energy() + calc_gs.get_reference_energy()

calc_exc = GPAW(mode='fd',
                h=h,
                txt='h2s_exc.txt',
                xc=xc,
                charge=q,
                spinpol=True,
                occupations=FermiDirac(0.0, fixmagmom=True),
                setups={0: '2p1ch'})

atoms[0].magmom = -q

atoms.calc = calc_exc
e_exc = atoms.get_potential_energy() + calc_exc.get_reference_energy()

with paropen('dks.result', 'w') as fd:
    print('Energy difference:', e_exc - e_gs, file=fd)
