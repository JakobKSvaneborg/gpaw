import pytest
import numpy as np
import subprocess
import sys

from ase.build import molecule

from gpaw.new.ase_interface import GPAW
from gpaw.gpu import cupy_is_fake


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
            'GPAW_NO_C_EXTENSION=1 GPAW_CPUPY=1 '
            f'python {__file__} {dtype} {gpu}',
            shell=True, capture_output=True,
            text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        print(e.stderr)
        raise e
    print(result.stdout)


@pytest.mark.serial
@pytest.mark.gpu
@pytest.mark.skipif(cupy_is_fake, reason='No cupy')
@pytest.mark.parametrize('dtype',
                         [np.complex128,
                          np.complex64,
                          np.float64,
                          np.float32])
def test_single_precision_gpu(dtype):
    run_single_precision(dtype=dtype, gpu='True')


def run_single_precision(dtype, gpu):
    atoms = molecule('H2O')
    atoms.center(vacuum=2.5)

    gpu = gpu == 'True'

    atoms.calc = GPAW(xc={'name': 'LDA'},
                      symmetry='off',
                      random=True,
                      convergence={'energy': 1e-5,
                                   'forces': 1e-3},
                      mode={'name': 'pw',
                            'ecut': 200.0,
                            'dtype': dtype},
                      parallel={'gpu': gpu}
                      )

    e_pot = atoms.get_potential_energy()
    expected_e = 9.595593485742606

    assert atoms.calc.wfs.dtype == dtype

    assert e_pot == pytest.approx(expected_e, rel=1e-3)


if __name__ == '__main__':
    dtypes = {'np.float32': np.float32,
              'np.float64': np.float64,
              'np.complex64': np.complex64,
              'np.complex128': np.complex128}
    run_single_precision(dtype=dtypes[sys.argv[1]], gpu=sys.argv[2])
