# Takes 30 mins with 8 cores in domain-parallel (memory ~500mb, max 1.5gb)

import numpy as np
from ase import Atom, Atoms

from gpaw import GPAW, MixerDif
from gpaw.mpi import world
from gpaw.lrtddft import LrTDDFT
from gpaw.pes.dos import DOSPES
from gpaw.pes.tddft import TDDFTPES

atoms = Atoms(
    [Atom('O', (7.999995107962541, 7.999996875878646, 8.298426499995756)),
     Atom('H', (8.000003614573515, 8.768044441257475, 7.699340728866722)),
     Atom('H', (8.000001513686263, 7.231958190201982, 7.699335817993486))])
atoms.center(vacuum=8)

h = 0.15
N_c = np.round(atoms.get_cell().diagonal() / h / 16) * 16

calc_params = dict(
    mode='fd',
    gpts=N_c,
    mixer=MixerDif(0.1, 5, weight=100.0),
    parallel={'domain': world.size},
    xc='PBE',
    spinpol=True,
    legacy_gpaw=True)

m_calc = GPAW(**calc_params, nbands=4, txt='H2O-m.txt')

m = atoms.copy()
m.set_initial_magnetic_moments([-1, 1, -1])
m.calc = m_calc
m.get_potential_energy()

d_calc = GPAW(**calc_params,
              nbands=16,
              convergence={'bands': 10},
              charge=1,
              txt='H2O-d.txt')

d = atoms.copy()
d.set_initial_magnetic_moments([-1, 0.5, -0.5])
d.calc = d_calc
d.get_potential_energy()

istart = 0  # band index of the first occ. band to consider
jend = 15   # band index of the last unocc. band to consider
d_lr = LrTDDFT(d_calc, xc='PBE', nspins=2,
               restrict={'istart': istart, 'jend': jend})
d_lr.diagonalize()

pes = TDDFTPES(m_calc, d_lr, d_calc)
pes.save_folded_pes('H2O-td.dat', folding=None)

pes = DOSPES(m_calc, d_calc, shift=True)
pes.save_folded_pes('H2O-dos.dat', folding=None)
