import pytest
import numpy as np
import subprocess

from ase.build import molecule

from gpaw.new.ase_interface import GPAW


@pytest.mark.serial
def test_single_precision():
    result = subprocess.run(
        f'GPAW_NO_C_EXTENSION=1 python {__file__}',
        shell=True, capture_output=True,
        text=True, check=True)
    result.stderr


def run_single_precision():
    atoms = molecule('H2')
    atoms.center(vacuum=2.5)
    atoms2 = atoms.copy()

    atoms.calc = GPAW(xc='PPLDA',
                      symmetry='off',
                      random=True,
                      mode={'name': 'pw',
                            'ecut': 200.0,
                            'dtype': complex})
    e_pot1 = atoms.get_potential_energy()

    atoms2.calc = GPAW(xc='PPLDA',
                       symmetry='off',
                       random=True,
                       mode={'name': 'pw',
                             'ecut': 200.0,
                             'dtype': np.float32})
    e_pot2 = atoms2.get_potential_energy()

    assert not atoms.calc.wfs.dtype == np.float32
    assert atoms2.calc.wfs.dtype == np.float32

    assert e_pot2 == pytest.approx(e_pot1, rel=1e-4)


if __name__ == '__main__':
    run_single_precision()
