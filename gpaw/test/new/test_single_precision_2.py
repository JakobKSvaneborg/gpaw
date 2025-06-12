import pytest
import numpy as np
import subprocess
import sys

from ase.build import molecule

from gpaw.new.ase_interface import GPAW
from gpaw.gpu import cupy_is_fake
from gpaw.mixer import FFTMixer


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
    from ase.build import mx2
    atoms = mx2('TaSe2', a=3.4)
    atoms2 = atoms.copy()
    atoms2.positions[:, 2] += 3.5 + 0.4
    atoms = atoms + atoms2
    atoms = atoms.repeat((2, 2, 1))
    atoms.center(axis=2, vacuum=6)

    gpu = gpu == 'True'

    atoms.calc = GPAW(xc={'name': 'PBE'},
                      symmetry='off',
                      random=True,
                      nbands=200,
                      convergence={'maximum iterations': 80,
                                   'eigenstates': 1e-8},
                      mode={'name': 'pw',
                            'ecut': 400.0,
                            'dtype': dtype},
                      mixer=FFTMixer(0.1),
                      eigensolver={'name': 'not-dav',
                                   'niter': 5},
                      occupations={'name': 'fermi-dirac',
                                   'width': 0.05},
                      parallel={'gpu': gpu}
                      )
    atoms.get_potential_energy()
    
    return

    atoms.calc.dft.params.convergence = {'energy': 1e-5,
                                         'density': 1e-6,
                                         'eigenstates': 5e-8,
                                         'eigenvalues': 5e-4,
                                         'forces': 5e-4}
    atoms.calc.dft.params.eigensolver = {'name': 'rmm-diis',
                                         'niter': 3,
                                         'trial_step': 0.1}
    atoms.calc.create_new_calculation_from_old(atoms)
    e_pot = atoms.get_potential_energy()

    expected_e = 9.595593485742606

    assert atoms.calc.wfs.dtype == dtype

    assert e_pot == pytest.approx(expected_e, rel=1e-3), e_pot - expected_e


if __name__ == '__main__':
    dtypes = {'np.float32': np.float32,
              'np.float64': np.float64,
              'np.complex64': np.complex64,
              'np.complex128': np.complex128}
    run_single_precision(dtype=dtypes[sys.argv[1]], gpu=sys.argv[2])
