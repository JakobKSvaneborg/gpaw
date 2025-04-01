from gpaw.solvation.sjm import SJM, SJMPower12Potential
from gpaw import FermiDirac
from ase.data.vdw import vdw_radii
from gpaw.solvation import (
    EffectivePotentialCavity,
    LinearDielectric,
    GradientSurface,
    SurfaceInteraction
)
from ase.build import fcc111


# Solvent parameters
u0 = 0.180
epsinf = 78.36
gamma = 0.00114843767916
T = 298.15

atoms = fcc111('H', size=(1, 1, 1), a=2.5)
atoms.center(axis=2, vacuum=5)
atoms.cell[2][2] = 10


def atomic_radii(atoms):
    return [vdw_radii[n] for n in atoms.numbers]


def calculator():
    return SJM(
        sj={'target_potential': 4.5,
            'jelliumregion': {'top': 10.},
            'tol': 0.5},
        gpts=(8, 8, 32),
        poissonsolver={'dipolelayer': 'xy'},
        kpts=(4, 4, 1),
        xc='PBE',
        spinpol=False,
        maxiter=1000,
        occupations=FermiDirac(0.1),
        convergence={'energy': 1,
                     'density': 1.0,
                     'eigenstates': 4.0,
                     'bands': 'occupied',
                     'forces': float('inf'),
                     'work function': 1},
        mode='lcao',
        basis='dzp',
        cavity=EffectivePotentialCavity(
            effective_potential=SJMPower12Potential(atomic_radii, u0),
            temperature=T,
            surface_calculator=GradientSurface()),
        dielectric=LinearDielectric(epsinf=epsinf),
        interactions=[SurfaceInteraction(surface_tension=gamma)])
