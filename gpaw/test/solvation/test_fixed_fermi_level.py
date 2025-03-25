import numpy as np
import pytest
from ase import Atoms
from gpaw import FermiDirac
from gpaw.new.ase_interface import GPAW


@pytest.mark.parametrize('mode', ['pw', 'fd'])
def test_ffl(mode):
    a = 1.4
    atoms = Atoms('H', cell=[a, a, 11.0], pbc=(1, 1, 0))
    atoms.positions[0, 2] = 4.0
    k = 2
    atoms.calc = GPAW(
        mode=mode,
        kpts=(k, k, 1),
        occupations=FermiDirac(0.2),
        poissonsolver={'dipolelayer': 'xy',
                       'zero_vacuum': True},
        background_charge=dict(charge=0.0001,
                               z1=7.0, z2=9.0,
                               workfunction=3.15))
    atoms.get_potential_energy()
    assert atoms.calc.get_fermi_level() == pytest.approx(-3.15, abs=0.001)
    if 0:
        v = atoms.calc.get_electrostatic_potential()
        import matplotlib.pyplot as plt
        plt.plot(np.linspace(0, 11, v.shape[2], 0), v[0, 0])
        plt.show()


if __name__ == '__main__':
    import sys
    test_ffl(sys.argv[1])
