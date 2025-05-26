from gpaw import GPAW
from ase.build import molecule
from ase.units import mol, kJ, kcal, Pascal, m
from ase.parallel import parprint
from gpaw.solvation import (
    SolvationGPAW,             # the solvation calculator
    EffectivePotentialCavity,  # cavity using an effective potential
    Power12Potential,          # a specific effective potential
    LinearDielectric,  # rule to construct permittivity func from the cavity
    GradientSurface,  # rule to calculate the surface area from the cavity
    SurfaceInteraction  # rule to calculate non-electrostatic interactions
)

# all parameters on the user side of the solvation API follow the ASE
# unit conventions (eV, Angstrom, ...)

# non-solvent related DFT parameters
h = 0.2
vac = 5.0

# solvent parameters for water from J. Chem. Phys. 141, 174108 (2014)
u0 = 0.180  # eV
epsinf = 78.36  # dimensionless
gamma = 18.4 * 1e-3 * Pascal * m  # convert from dyne / cm to eV / Angstrom**2
T = 298.15  # Kelvin
atomic_radii = {'H': 1.09}


# create Atoms object for ethanol and add vacuum
atoms = molecule('CH3CH2OH')
atoms.center(vacuum=vac)

# perform gas phase calculation
atoms.calc = GPAW(mode='fd', xc='PBE', h=h, txt='gasphase.txt')
Egasphase = atoms.get_potential_energy()

# perform calculation with continuum solvent model from
# J. Chem. Phys. 141, 174108 (2014)
atoms.calc = SolvationGPAW(
    mode='fd', xc='PBE', h=h, txt='water.txt',
    cavity=EffectivePotentialCavity(
        effective_potential=Power12Potential(atomic_radii, u0),
        temperature=T,
        surface_calculator=GradientSurface()),
    dielectric=LinearDielectric(epsinf=epsinf),
    interactions=[SurfaceInteraction(surface_tension=gamma)])
Ewater = atoms.get_potential_energy()

# calculate solvation Gibbs energy in various units
DGSol_eV = Ewater - Egasphase
DGSol_kJ_per_mol = DGSol_eV / (kJ / mol)
DGSol_kcal_per_mol = DGSol_eV / (kcal / mol)

parprint('calculated Delta Gsol = %.0f meV = %.1f kJ / mol = %.1f kcal / mol' %
         (DGSol_eV * 1000., DGSol_kJ_per_mol, DGSol_kcal_per_mol))
