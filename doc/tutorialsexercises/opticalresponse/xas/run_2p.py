from ase.build import molecule

from gpaw import GPAW, setup_paths
from gpaw.utilities.adjust_cell import adjust_cell
from gpaw.xas import XAS

setup_paths.insert(0, '.')

box = 7
h = 0.2
xc = 'PBE'
atoms = molecule('SH2')
adjust_cell(atoms, box, h)

calc = GPAW(mode='fd',
            nbands=-30,
            h=h,
            txt='h2s_xas.txt',
            setups={'S': '2p05ch'},
            xc=xc)
# the number of unoccupied stated will determine how
# high you will get in energy

atoms.calc = calc
atoms.get_potential_energy()

calc.write('h2s_xas.gpw')

# write out the marix ellement
xas = XAS(calc)

xas.write('me_h2s_xas.npz')
