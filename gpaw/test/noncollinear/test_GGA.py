import pytest
from ase import Atoms

from gpaw import GPAW
from gpaw.mpi import size


def test_noncollinear_o2(gpaw_new):
    a = Atoms('H', [[0, 0, 0]], magmoms=[1])
    a.center(vacuum=2.5)
    a.calc = GPAW(mode='pw',
                  xc='PBE',
                  symmetry='off',
                  txt=None,
                  experimental={'magmoms': [[0, 0.5, 0.5]]})
    with pytest.raises(ValueError):
        a.get_potential_energy()
