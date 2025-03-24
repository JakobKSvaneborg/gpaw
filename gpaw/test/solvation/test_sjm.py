from ase.build import fcc111
from ase.data.vdw import vdw_radii
from gpaw import FermiDirac
from gpaw.new.sjm import SJM
from gpaw.solvation import (EffectivePotentialCavity, GradientSurface,
                            LinearDielectric, SurfaceInteraction)
from gpaw.solvation.sjm import SJM as OldSJM, SJMPower12Potential


def test_sjm(gpaw_new):
    # Solvent parameters
    u0 = 0.180  # eV
    epsinf = 78.36  # Dielectric constant of water at 298 K
    gamma = 0.00114843767916  # 18.4*1e-3 * Pascal* m
    T = 298.15   # K

    def atomic_radii(atoms):
        return [vdw_radii[n] for n in atoms.numbers]

    # Structure is created
    atoms = fcc111('Au', size=(1, 1, 3))
    atoms.cell[2][2] = 15
    atoms.translate([0, 0, 6 - min(atoms.positions[:, 2])])

    # SJM parameters
    potential = 4.5
    tol = 0.02
    sj = {'target_potential': potential,
          'excess_electrons': -0.045,
          'jelliumregion': {'top': 14.5},
          'tol': tol}

    convergence = {
        'energy': 0.05 / 8.,
        'density': 1e-4,
        'eigenstates': 1e-4}

    params = dict(
        mode='fd',
        gpts=(8, 8, 48),
        kpts=(2, 2, 1),
        xc='PBE',
        convergence=convergence,
        occupations=FermiDirac(0.1))

    solvation = dict(
        cavity=EffectivePotentialCavity(
            effective_potential=SJMPower12Potential(atomic_radii, u0),
            temperature=T,
            surface_calculator=GradientSurface()),
        dielectric=LinearDielectric(epsinf=epsinf),
        interactions=[SurfaceInteraction(surface_tension=gamma)])

    if not gpaw_new:
        calc = OldSJM(**params, sj=sj, **solvation)
    else:
        calc = GPAW(
            **params,
            envoronment=SJM(*sj, *solvation)))

    # Run the calculation
    atoms.calc = calc
    atoms.get_potential_energy()
    assert abs(calc.get_electrode_potential() - potential) < tol
