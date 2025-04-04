import numpy as np
from ase import Atoms
from gpaw import FermiDirac
from gpaw.new.ase_interface import GPAW
from gpaw.new.sjm import SJM
from gpaw.solvation import (EffectivePotentialCavity, GradientSurface,
                            LinearDielectric, SurfaceInteraction)
from gpaw.solvation.sjm import SJM as OldSJM, SJMPower12Potential
from ase.data.vdw import vdw_radii


def test_h(gpaw_new, mode):
    u0 = 0.180  # eV
    gamma = 0.00114843767916  # 18.4*1e-3 * Pascal* m
    T = 298.15   # K

    def atomic_radii(atoms):
        return [vdw_radii[n] for n in atoms.numbers]

    a = 1.4
    atoms = Atoms('H', cell=[a, a, 11.0], pbc=(1, 1, 0))
    atoms.positions[0, 2] = 4.0
    k = 2

    params = dict(
        mode=mode,
        kpts=(k, k, 1),
        occupations=FermiDirac(0.2))
    solvation = dict(
        cavity=EffectivePotentialCavity(
            effective_potential=SJMPower12Potential(atomic_radii, u0),
            temperature=T,
            surface_calculator=GradientSurface()),
        dielectric=LinearDielectric(epsinf=1.0),
        interactions=[SurfaceInteraction(surface_tension=gamma)])
    sjm = {'target_potential': 7.5,
           'excess_electrons': -0.0045,
           'jelliumregion': {'top': 9.0, 'bottom': 7.0},
           'tol': 0.01}

    if gpaw_new:
        atoms.calc = GPAW(environment=SJM(**solvation, **sjm), **params)
    else:
        atoms.calc = OldSJM(**solvation, sj=sjm, **params)
    atoms.get_potential_energy()
    # assert atoms.calc.get_fermi_level() == pytest.approx(-3.15, abs=0.001)
    if 1:
        v = atoms.calc.get_electrostatic_potential()
        import matplotlib.pyplot as plt
        plt.plot(np.linspace(0, 11, v.shape[2], 0), v[0, 0])
        plt.show()


if __name__ == '__main__':
    test_h(1, 'pw')
