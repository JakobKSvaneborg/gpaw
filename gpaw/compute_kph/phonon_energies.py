from ase.build import bulk
from ase.calculators.emt import EMT
from ase.phonons import Phonons
from gpaw import GPAW
from gpaw.mpi import world

from ase.build import bulk
from gpaw import GPAW,PW, FermiDirac
from gpaw.mixer import Mixer
import numpy as np

calc = GPAW('scf_PW.gpw')
atoms = calc.atoms
N = 4
ph = Phonons(atoms, calc, supercell=(N, N, N), delta=0.05)
ph.run()
ph.read(method='frederiksen', acoustic=True)
#ph.clean()

path = atoms.cell.bandpath('GXWKGLUWL', npoints=100)
if world.rank == 0:
    print('made the path', flush=True)

q = np.load('q_list.npy')
w = ph.band_structure(q, modes=False, born=False, verbose=True)
if world.rank == 0:
    print('made w', flush=True)

q_indices = np.load('q_indices.npy')

np.save('w_phonon_unique_q.npy', w)

w_full = w[q_indices, ...]

np.save('phonon_energies_all.npy', w_full)

#import matplotlib.pyplot as plt  # noqa

#fig = plt.figure(figsize=(7, 4))
#ax = fig.add_axes([0.12, 0.07, 0.67, 0.85])

#emax = 0.035
#bs.plot(ax=ax, emin=0.0, emax=emax)
#fig.savefig('Al_phonon.png')

