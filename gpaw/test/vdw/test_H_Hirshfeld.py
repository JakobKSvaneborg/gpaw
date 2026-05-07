"""Test Hirshfeld for spin/no spin consistency."""
import pytest
from ase import Atoms
from ase.parallel import parprint

from gpaw import GPAW, FermiDirac
from gpaw.analyse.hirshfeld import HirshfeldPartitioning


def test_vdw_H_Hirshfeld():
    h = 0.25
    box = 3

    atoms = Atoms('H')
    atoms.center(vacuum=box)

    volumes = []
    for spinpol in [False, True]:
        calc = GPAW(legacy_gpaw=True,
                    mode='fd',
                    h=h,
                    occupations=FermiDirac(0.1, fixmagmom=spinpol),
                    convergence={'density': 1e-6},
                    spinpol=spinpol)
        calc.calculate(atoms)
        vol = HirshfeldPartitioning(calc).get_effective_volume_ratios()
        volumes.append(vol)
    parprint(volumes)
    assert volumes[0][0] == pytest.approx(volumes[1][0], abs=2e-7)
