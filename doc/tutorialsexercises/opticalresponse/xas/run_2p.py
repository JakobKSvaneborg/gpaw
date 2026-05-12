from ase.build import molecule

from gpaw import GPAW, setup_paths
from gpaw.utilities.adjust_cell import adjust_cell
from gpaw.xas import XAS

setup_paths.insert(0, '.')

box = 3
h = 0.25
xc = 'PBE'
atoms = molecule('SH2')
adjust_cell(atoms, box, h)

calc = GPAW(mode='fd',
            # the number of unoccupied stated will determine how
            # high you will get in energy
            nbands=-30,
            h=h,
            txt='h2s_xas.txt',
            setups={'S': 'hch2p'},
            xc=xc,
            legacy_gpaw=True)

atoms.calc = calc
atoms.get_potential_energy()

xas = XAS(calc)
# write out the marix elements
xas.write('me_h2s_xas.npz')
