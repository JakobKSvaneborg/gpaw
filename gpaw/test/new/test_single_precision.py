import pytest
import numpy as np
import subprocess
import sys

from ase.build import molecule
from ase.build import bulk

from gpaw.new.ase_interface import GPAW


@pytest.mark.serial
@pytest.mark.parametrize('dtype',
                         ['np.complex128',
                          'np.complex64',
                          'np.float64',
                          'np.float32'])
@pytest.mark.parametrize('gpu', [False, True])
def test_single_precision(dtype, gpu):
    try:
        result = subprocess.run(
            f'GPAW_NO_C_EXTENSION=1 python {__file__} {dtype} {gpu}',
            shell=True, capture_output=True,
            text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        print(e.stderr)
        raise e
    print(result.stdout)


def run_single_precision(dtype, gpu):
    #atoms = molecule('H20')
    atoms = molecule('C60')
    atoms.center(vacuum=2.5)
    atoms = atoms.repeat((2, 1, 1))
    #atoms = bulk('Cu')

    gpu = gpu == 'True'

    atoms.calc = GPAW(xc={'name': 'LDA'},
                      symmetry='off',
                      random=True,
                      convergence={'energy': 1e-5,
                                   'eigenstates': 1e-6,
                                   'density': 1e-5},
                      #kpts={'density': 1},
                      mode={'name': 'pw',
                            'ecut': 600.0,
                            'dtype': dtype},
                      parallel={'gpu': gpu}
                      )

    e_pot = atoms.get_potential_energy()
    #expected_e = 9.595593485742606
    #expected_e = -2.19724921704334
    #expected_e = -15.047246
    #expected_e = -582.809148
    expected_e = -1165.754498
    assert atoms.calc.wfs.dtype == dtype

    assert e_pot == pytest.approx(expected_e, rel=1e-3), e_pot - expected_e


if __name__ == '__main__':
    dtypes = {'np.float32': np.float32,
              'np.float64': np.float64,
              'np.complex64': np.complex64,
              'np.complex128': np.complex128}
    run_single_precision(dtype=dtypes[sys.argv[1]], gpu=sys.argv[2])
