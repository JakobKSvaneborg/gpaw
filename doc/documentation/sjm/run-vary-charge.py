import pickle
import numpy as np
from ase import Atoms
from ase.units import Pascal, m
from ase.parallel import paropen
from gpaw.solvation.sjm import SJM, SJMPower12Potential
from gpaw import FermiDirac
from gpaw.solvation import (
    EffectivePotentialCavity,
    LinearDielectric,
    GradientSurface,
    SurfaceInteraction
)


def write_everything(calc, label):
    """Writes out all the useful data after the SJM calculation."""
    atoms.write(f'atoms{label}.traj')
    esp = atoms.calc.get_electrostatic_potential()
    with paropen(f'esp{label}.pckl', 'wb') as f:
        pickle.dump(esp, f)
    n = calc.get_all_electron_density()
    with open(f'allelectrondensity{label}.pckl', 'wb') as f:
        pickle.dump(n, f)


def log_potential(calc):
    """Saves charge and potential in simple text file for subsequent
    plot."""
    with paropen('potential.txt', 'a') as f:
        f.write('{:10.4f}, {:10.4f}\n'
                .format(calc.results['electrode_potential'],
                        calc.results['excess_electrons']))


atoms = Atoms(symbols='Pt27OH2OH2OH2OH2OH2OH2',
              pbc=np.array([True, True, False]),
              cell=np.array(
                  [[8.42164176, 0.00000000, 0.00000000],
                   [4.21082088, 7.29335571, 0.00000000],
                   [0.00000000, 0.00000000, 30.00000000]]),
              positions=np.array(
                  [[1.40360696, 0.81037286, 6.65200000],
                   [4.21082088, 0.81037286, 6.65200000],
                   [7.01803480, 0.81037286, 6.65200000],
                   [2.80721392, 3.24149143, 6.65200000],
                   [5.61442784, 3.24149143, 6.65200000],
                   [8.42164176, 3.24149143, 6.65200000],
                   [4.21082088, 5.67261000, 6.65200000],
                   [7.01803480, 5.67261000, 6.65200000],
                   [9.82524872, 5.67261000, 6.65200000],
                   [0.03036456, 1.64159455, 8.98780067],
                   [2.83952601, 1.63656373, 8.96913616],
                   [5.65700108, 1.63548404, 8.98839035],
                   [1.43694079, 4.06184422, 8.97011747],
                   [4.25013151, 4.06794136, 8.97319238],
                   [7.04795258, 4.05874015, 8.98685186],
                   [2.83797345, 6.50356945, 8.97451149],
                   [5.64860418, 6.49353012, 8.97149139],
                   [8.46007009, 6.50339758, 8.97110938],
                   [0.04167726, 0.01659389, 11.33367526],
                   [2.84872686, 0.01609218, 11.33191349],
                   [5.65812910, 0.01711834, 11.39819957],
                   [1.45292097, 2.44351232, 11.38081839],
                   [4.25298704, 2.44406661, 11.32946091],
                   [7.06279504, 2.44664945, 11.44031710],
                   [2.84886720, 4.87885276, 11.32743866],
                   [5.65767156, 4.88199267, 11.38959530],
                   [8.46251173, 4.88143235, 11.31836346],
                   [1.39280453, 2.54002304, 14.97410114],
                   [1.91436258, 3.34474212, 15.30247756],
                   [1.60746426, 2.47760171, 14.01227886],
                   [2.81907167, -0.18784312, 15.85479905],
                   [3.75807226, -0.09412171, 15.55917086],
                   [2.37285739, 0.64492180, 15.59351972],
                   [5.59285245, -0.22157210, 15.07552505],
                   [6.07745845, 0.54995965, 15.43008777],
                   [5.64779097, -0.12087038, 14.09724225],
                   [7.28795735, 2.71603215, 14.82288807],
                   [8.34375424, 2.73525339, 14.98389958],
                   [6.79086560, 3.65674631, 14.98624863],
                   [5.91582226, 4.82631629, 14.97807974],
                   [6.34845728, 5.68351739, 15.30313705],
                   [5.73153676, 4.98072249, 14.02008620],
                   [2.76657910, 4.66918882, 15.85175518],
                   [3.71162532, 4.72529333, 15.60227465],
                   [2.36035150, 5.52417539, 15.56370623]]))

# Solvated jellium parameters.
sj = {'excess_electrons': 0.}

# Implicit solvent parameters (to SolvationGPAW).
epsinf = 78.36  # dielectric constant of water at 298 K
gamma = 18.4 * 1e-3 * Pascal * m
cavity = EffectivePotentialCavity(
    effective_potential=SJMPower12Potential(H2O_layer=True),
    temperature=298.15,  # K
    surface_calculator=GradientSurface())
dielectric = LinearDielectric(epsinf=epsinf)
interactions = [SurfaceInteraction(surface_tension=gamma)]

atoms.calc = SJM(
    mode='fd',
    txt='gpaw-charge.txt',
    kpts=(4, 4, 1),
    gpts=(48, 48, 192),
    xc='PBE',
    occupations=FermiDirac(0.1),
    convergence={'work function': 0.001},
    # Solvated jellium parameters.
    sj=sj,
    # Implicit solvent parameters.
    cavity=cavity,
    dielectric=dielectric,
    interactions=interactions)

for excess_electrons in [-0.2, -0.1, 0., .1, .2]:
    sj['excess_electrons'] = excess_electrons
    atoms.calc.set(sj=sj)
    atoms.get_potential_energy()
    log_potential(atoms.calc)
    write_everything(atoms.calc, label='{:.4f}'.format(excess_electrons))
