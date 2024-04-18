from ase.vibrations.resonant_raman import ResonantRamanCalculator
from ase.io import read

from gpaw import GPAW, FermiDirac
from gpaw.utilities.adjust_cell import adjust_cell
from gpaw.lrtddft import LrTDDFT

h = 0.25
xc = 'PBE'
atoms = read('relaxed.traj')
adjust_cell(atoms, 4., h=h)

atoms.calc = GPAW(mode='fd', xc=xc, h=h, nbands=50,
                  occupations=FermiDirac(width=0.1),
                  eigensolver='cg', symmetry={'point_group': False},
                  convergence={'eigenstates': 1.e-5, 'bands': -10})
atoms.get_potential_energy()

erange = 17
ext = '_erange{0}'.format(erange)
gsname = exname = 'rraman' + ext
rr = ResonantRamanCalculator(
    atoms, LrTDDFT, name=gsname, exname=exname,
    exkwargs={'restrict': {'energy_range': erange, 'eps': 0.4}},)
rr.run()
