import pytest
import numpy as np
import subprocess

from ase.build import molecule
from ase.build import bulk

from gpaw.new.ase_interface import GPAW


@pytest.mark.serial
def test_single_precision():
    try:
        result = subprocess.run(
            f'GPAW_NO_C_EXTENSION=1 python {__file__}',
            shell=True, capture_output=True,
            text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        print(e.stderr)
        raise e
    print(result.stdout)


def run_single_precision():
    atoms = bulk('Au')
    atoms.center(vacuum=2.5)
    atoms2 = atoms.copy()

    atoms.calc = GPAW(xc={'name': 'LDA'},
                      symmetry='off',
                      random=True,
                      kpts={'density': 2},
                      mode={'name': 'pw',
                            'ecut': 200.0,
                            'dtype': np.complex64},
                     parallel={'gpu': True}
                     )
    e_pot1 = atoms.get_potential_energy()

    atoms2.calc = GPAW(xc={'name': 'LDA'},
                       symmetry='off',
                       random=True,
                       kpts={'density': 2},
                       mode={'name': 'pw',
                             'ecut': 200.0,
                             'dtype': np.complex128},
                       parallel={'gpu': True}
                       )
    e_pot2 = atoms2.get_potential_energy()

    assert atoms.calc.wfs.dtype == np.complex64
    assert atoms2.calc.wfs.dtype == np.complex128

    assert e_pot2 == pytest.approx(e_pot1, rel=1e-3)


if __name__ == '__main__':
    run_single_precision()
