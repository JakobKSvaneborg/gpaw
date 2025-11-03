import numpy as np

from ase.optimize import BFGS
from ase import Atoms

from gpaw import restart
from gpaw.solvation.sjm import SJM, SJMPower12Potential
from gpaw import FermiDirac
from gpaw.solvation import (
    EffectivePotentialCavity,
    LinearDielectric,
    GradientSurface,
    SurfaceInteraction)


# Solvent parameters
u0 = 0.180  # eV
epsinf = 78.36  # dielectric constant of water at 298 K
gamma = 0.00114843767916  # 18.4*1e-3 * Pascal*m
T = 298.15    # K

atoms = Atoms(symbols='Au6OH2OH2OH2OH2',
              pbc=[True, True, False],
              cell=[8.867119036079306, 5.119433562416841, 15.0])
atoms.set_positions(
    np.array([[0.56164375, 0.39256423, 5.4220237],
              [3.51680174, 0.39253274, 5.42219168],
              [6.4732849, 0.39549906, 5.42907972],
              [2.03914701, 2.95449556, 5.42846579],
              [4.99533678, 2.95180933, 5.42125025],
              [7.9504729, 2.95176062, 5.42333033],
              [6.58498136, 0.64530367, 9.09028872],
              [7.06181382, -0.19870328, 9.30204832],
              [6.58872823, 0.67663023, 8.10927953],
              [7.99675219, 3.3136834, 9.76193639],
              [7.58790792, 2.45509461, 9.51090101],
              [8.93840968, 3.2284547, 9.49325933],
              [3.56839946, 0.71728602, 9.69844062],
              [3.16824247, -0.14379405, 9.43936165],
              [4.51456907, 0.64768502, 9.44106286],
              [2.14806966, 3.19083417, 9.07185988],
              [2.62667731, 2.34458289, 9.26969167],
              [2.10900492, 3.20816497, 8.09062402]]))

sj = {'target_potential': 4.5,
      'excess_electrons': 0.124,
      'jelliumregion': {'top': 14.},
      'tol': 0.005}

... = dict(
    cavity=EffectivePotentialCavity(
        effective_potential=SJMPower12Potential(u0=u0,
                                                H2O_layer=True),
        temperature=T,
        surface_calculator=GradientSurface()),
    dielectric=LinearDielectric(epsinf=epsinf),
    interactions=[SurfaceInteraction(surface_tension=gamma)])

calc = SJM(mode='fd',
           sj=sj,
           gpts=(48, 32, 88),
           kpts=(2, 2, 1),
           xc='PBE',
           occupations=FermiDirac(0.1),
           txt='sjm.txt',
           convergence={'energy': 0.000005})

atoms.calc = calc
E = []

for pot in [4.5, None, 4.3, 4.5]:
    if pot is None:
        calc.set(sj={'excess_electrons': 0.2, 'target_potential': None})
    else:
        calc.set(sj={'target_potential': pot})
    E.append(atoms.get_potential_energy())

    if pot is None:
        assert abs(calc.wfs.nvalence - calc.setups.nvalence - 0.2) < 1e-4
    else:
        assert abs(calc.get_electrode_potential() - pot) < 0.005

assert abs(E[0] - E[-1]) < 1e-2

calc.write('sjm.gpw')

atoms, calc = restart('sjm.gpw', Class=SJM)
calc.set(sj={'tol': 0.002})
atoms.get_potential_energy()
assert abs(calc.get_electrode_potential() - 4.5) < 0.002

calc.set(sj={'jelliumregion': {'top': 13}, 'tol': 0.01})
atoms.get_potential_energy()
assert abs(calc.get_electrode_potential() - 4.5) < 0.01

calc.set(sj={'jelliumregion': {'thickness': 2}})
atoms.get_potential_energy()
assert abs(calc.get_electrode_potential() - 4.5) < 0.01

qn = BFGS(atoms, maxstep=0.05)
qn.run(fmax=0.05, steps=2)
assert abs(calc.get_electrode_potential() - 4.5) < 0.01
