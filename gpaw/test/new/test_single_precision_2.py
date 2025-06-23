import pytest
import numpy as np
import subprocess
import sys

from gpaw.dft import RMMDIIS
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
    atoms = mx2('TaSe2', a=3.3)
    atoms2 = mx2('MoS2', a=3.3)
    atoms2.positions[:, 2] += 3.5 + 5
    atoms = atoms + atoms2
    atoms = atoms.repeat((1, 1, 1))
    atoms.center(axis=2, vacuum=5.5)

    gpu = gpu == 'True'

    atoms.calc = GPAW(xc={'name': 'PBE'},
                      symmetry='off',
                      random=True,
                      convergence={'maximum iterations': 200,
                                   'eigenstates': 1e-7},
                      mode={'name': 'pw',
                            'ecut': 600.0,
                            'dtype': dtype},
                      mixer=FFTMixer(0.1),
                      eigensolver={'name': 'not-dav',
                                   'niter': 3},
                      occupations={'name': 'fermi-dirac',
                                   'width': 0.05},
                      parallel={'gpu': gpu}
                      )
    atoms.get_potential_energy()
    return

    atoms.calc.dft.params.eigensolver = RMMDIIS(niter=2,
                                                trial_step=0.1)
    atoms.calc.dft.params.convergence = {'eigenstates': 1e-8,
                                         'maximum iterations': 100}
    atoms.calc.create_new_calculation_from_old(atoms)
    atoms.get_potential_energy()

    return
    atoms.calc = GPAW(xc={'name': 'PBE'},
                      symmetry='off',
                      convergence={'maximum iterations': 160,
                                   'eigenstates': 1e-80},
                      mode={'name': 'pw',
                            'ecut': 600.0,
                            'dtype': dtype},
                      mixer=FFTMixer(0.1),
                      eigensolver={'name': 'rmm-diis',
                                   'niter': 3,
                                   'trial_step': 0.1},
                      occupations={'name': 'fermi-dirac',
                                   'width': 0.05},
                      parallel={'gpu': gpu}
                      )
    atoms.get_potential_energy()


if __name__ == '__main__':
    dtypes = {'np.float32': np.float32,
              'np.float64': np.float64,
              'np.complex64': np.complex64,
              'np.complex128': np.complex128}
    run_single_precision(dtype=dtypes[sys.argv[1]], gpu=sys.argv[2])
