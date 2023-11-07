import ase.io
from ase.units import Pascal, m
from ase.optimize import BFGS
from ase.neb import interpolate
from ase.mep import DyNEB

from gpaw.solvation.sjm import SJM, SJMPower12Potential
from gpaw.solvation import (
    EffectivePotentialCavity,
    LinearDielectric,
    GradientSurface,
    SurfaceInteraction)


def make_calculator(index):
    # Solvated jellium parameters.
    sj = {'target_potential': 4.2,
          'tol': 0.01,
          'excess_electrons': -0.01232,  # guess from previous
          'slope': -50.}  # guess from previous
    # Implicit solvent parameters (to SolvationGPAW).
    epsinf = 78.36  # dielectric constant of water at 298 K
    gamma = 18.4 * 1e-3 * Pascal * m
    cavity = EffectivePotentialCavity(
        effective_potential=SJMPower12Potential(H2O_layer=True),
        temperature=298.15,  # K
        surface_calculator=GradientSurface())
    dielectric = LinearDielectric(epsinf=epsinf)
    interactions = [SurfaceInteraction(surface_tension=gamma)]
    calc = SJM(
        # General GPAW parameters.
        mode='fd',
        txt=f'gpaw-{index:d}.txt',
        gpts=(16, 16, 136),
        kpts=(9, 9, 1),
        xc='PBE',
        maxiter=1000,
        # Solvated jellium parameters.
        sj=sj,
        # Implicit solvent parameters.
        cavity=cavity,
        dielectric=dielectric,
        interactions=interactions)
    return calc


# Create the band of images, attaching a calc to each.
initial = ase.io.read('qn-Au111-H-sim-1.traj')
final = ase.io.read('qn-Au111-H-hollow-1.traj')
images = [initial]
for index in range(5):
    images += [initial.copy()]
    images[-1].calc = make_calculator(index + 1)
images += [final]
interpolate(images)

# Create and relax the DyNEB.
neb = DyNEB(images)
opt = BFGS(neb, logfile='neb.log', trajectory='neb.traj')
opt.run(fmax=0.05)
