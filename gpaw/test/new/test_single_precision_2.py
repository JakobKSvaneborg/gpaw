import pytest
import numpy as np
import subprocess
import sys
import os

# from gpaw.dft import RMMDIIS
from gpaw.new.ase_interface import GPAW
from gpaw.gpu import cupy_is_fake
from gpaw.mixer import FFTMixerFull


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
    atoms = mx2('MoFe2', a=3.14)
    # atoms2 = mx2('MoS2', a=3.3)
    # atoms2.positions[:, 2] += 3.5 + 5
    # atoms = atoms + atoms2
    atoms = atoms.repeat((1, 1, 1))
    atoms.center(axis=2, vacuum=5.5)
    # atoms.set_initial_magnetic_moments([1, ] * len(atoms))

    gpu = gpu == 'True'

    atoms.calc = GPAW(xc={'name': 'LDA'},
                      # kpts=(6, 6, 1),
                      # symmetry='off',
                      convergence={'maximum iterations': 300},
                      mode={'name': 'pw',
                            'ecut': 400.0,
                            'dtype': dtype},
                      mixer=FFTMixerFull(0.07),
                      poissonsolver={'fast': False},
                      eigensolver={'name': 'not-dav',
                                   'niter': 2,
                                   'include_CG': True},
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
    if os.environ.get('GPAW_TRACE') == '1':
        from gpaw.new.timer import global_timer
        from gpaw.utilities.timing import GPUProfiler
        with global_timer.context(GPUProfiler("gpu")) as timer:
            run_single_precision(dtype=dtypes[sys.argv[1]], gpu=sys.argv[2])
    else:
        run_single_precision(dtype=dtypes[sys.argv[1]], gpu=sys.argv[2])
