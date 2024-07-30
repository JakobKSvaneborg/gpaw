from ase import Atoms, io, optimize
from gpaw import GPAW, FermiDirac
from gpaw.utilities.adjust_cell import adjust_cell
from gpaw.lrtddft import LrTDDFT
from gpaw.lrtddft.excited_state import ExcitedState

box = 5.      # box dimension
h = 0.25      # grid spacing
width = 0.01  # Fermi width
nbands = 6    # bands in GS calculation
nconv = 4     # bands in GS calculation to converge
R = 2.99      # starting distance
iex = 1       # excited state index
d = 0.01      # step for numerical force evaluation
exc = 'LDA'   # xc for the linear response TDDFT kernel
parallel = 4  # number of independent parrallel workers for force evaluation

s = Atoms('Na2', [[0, 0, 0], [0, 0, R]])
adjust_cell(s, box, h=h)

c = GPAW(mode='fd', h=h, nbands=nbands, eigensolver='cg',
         occupations=FermiDirac(width=width),
         setups={'Na': '1'},
         convergence={'bands': nconv})
c.calculate(s)
lr = LrTDDFT(c, xc=exc, restrict={'eps': 0.1, 'jend': nconv - 1})

ex = ExcitedState(lr, iex, d=d, parallel=parallel)
s.calc = ex

ftraj = 'relax_ex' + str(iex)
ftraj += '_box' + str(box) + '_h' + str(h)
ftraj += '_d' + str(d) + '.traj'
traj = io.Trajectory(ftraj, 'w', s)
dyn = optimize.FIRE(s)
dyn.attach(traj.write)
dyn.run(fmax=0.05)
