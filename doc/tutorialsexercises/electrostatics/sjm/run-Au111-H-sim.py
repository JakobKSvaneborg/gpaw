from ase import Atom
import ase.io
from ase.units import Pascal, m
from ase.optimize import BFGS
from ase.constraints import FixAtoms

from gpaw.solvation.sjm import SJM, SJMPower12Potential
from gpaw.solvation import (
    EffectivePotentialCavity,
    LinearDielectric,
    GradientSurface,
    SurfaceInteraction)

# Add an H adsorbate.
atoms = ase.io.read('Au111.traj')
atoms.append(Atom('H', atoms[2].position + (0., 0., 1.5)))

# Fix some atoms.
atoms.set_constraint(FixAtoms(indices=[0, 1]))

# Solvated jellium parameters.
sj = {'target_potential': 4.2,
      'tol': 0.2,
      'always_adjust': True}

# Implicit solvent parameters (to SolvationGPAW).
epsinf = 78.36  # dielectric constant of water at 298 K
gamma = 18.4 * 1e-3 * Pascal * m
cavity = EffectivePotentialCavity(
    effective_potential=SJMPower12Potential(H2O_layer=True),
    temperature=298.15,  # K
    surface_calculator=GradientSurface())
dielectric = LinearDielectric(epsinf=epsinf)
interactions = [SurfaceInteraction(surface_tension=gamma)]

# The calculator
calc = SJM(
    # General GPAW parameters.
    mode='fd',
    txt='Au111-H-sim.txt',
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
atoms.calc = calc

# Run the calculation.
opt = BFGS(atoms, trajectory='qn-Au111-H-sim.traj',
           logfile='qn-Au111-H-sim.log')
opt.run()

# Tighten the tolerances again.
sj['tol'] = 0.01
sj['always_adjust'] = False
sj['slope'] = None
calc.set(sj=sj)
opt = BFGS(atoms, trajectory='qn-Au111-H-sim-1.traj',
           logfile='qn-Au111-H-sim.log')
opt.run()
