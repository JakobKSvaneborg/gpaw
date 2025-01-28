import pytest
import numpy as np

from ase.build import molecule

from gpaw.new.ase_interface import GPAW


def test_single_precision():
    atoms = molecule('H2')
    atoms.center(vacuum=2.5)
    atoms2 = atoms.copy()

    atoms.calc = GPAW(xc='PPLDA',
                      symmetry='off',
                      random=True,
                      mode={'name': 'pw',
                            'ecut': 200.0,
                            'dtype': np.complex128})
    e_pot1 = atoms.get_potential_energy()

    atoms2.calc = GPAW(xc='PPLDA',
                       symmetry='off',
                       random=True,
                       mode={'name': 'pw',
                             'ecut': 200.0,
                             'dtype': np.float32})
    e_pot2 = atoms2.get_potential_energy()

    assert e_pot2 == pytest.approx(e_pot1, rel=1e-4)
