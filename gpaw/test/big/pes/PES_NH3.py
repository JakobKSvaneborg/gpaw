# Takes 30 mins with 8 cores in domain-parallel (memory ~500mb, max 1.5gb)

import numpy as np
from ase import Atom, Atoms

from gpaw import GPAW, MixerDif, mpi
from gpaw.lrtddft import LrTDDFT
from gpaw.pes.dos import DOSPES
from gpaw.pes.tddft import TDDFTPES

atoms = Atoms(
    [Atom('N', (5.915080000001099, 5.584099935483791, 5.517093011559592)),
     Atom('H', (5.915080000000824, 6.534628192918412, 5.154213005717621)),
     Atom('H', (6.739127072810322, 5.112845817889995, 5.151626633318725)),
     Atom('H', (5.091032927190902, 5.112845817890320, 5.151626633319492))])
atoms.center(vacuum=8)

h = 0.15
N_c = np.round(atoms.get_cell().diagonal() / h / 16) * 16

calc_params = dict(
    mode='fd',
    gpts=N_c,
    mixer=MixerDif(0.1, 5, weight=100.0),
    parallel={'domain': mpi.size},
    xc='PBE',
    spinpol=True)

m_calc = GPAW(**calc_params, nbands=4, txt='NH3-m.txt')

m = atoms.copy()
m.set_initial_magnetic_moments([-1, 1, 1, -1])
m.calc = m_calc
m.get_potential_energy()

d_calc = GPAW(**calc_params,
              nbands=16,
              charge=1,
              txt='NH3-d.txt')

d = atoms.copy()
d.set_initial_magnetic_moments([-1, 0.5, 0.5, -0.5])
d.calc = d_calc
d.get_potential_energy()

istart = 0  # band index of the first occ. band to consider
jend = 15   # band index of the last unocc. band to consider
d_lr = LrTDDFT(d_calc, xc='PBE', nspins=2,
               restrict={'istart': istart, 'jend': jend})
d_lr.diagonalize()

pes = TDDFTPES(m_calc, d_lr, d_calc)
pes.save_folded_pes('NH3-td.dat', folding=None)

pes = DOSPES(m_calc, d_calc, shift=True)
pes.save_folded_pes('NH3-dos.dat', folding=None)
