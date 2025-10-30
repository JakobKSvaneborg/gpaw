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
def test_read_write_gpw(in_tmp_dir, dtype, gpu):
    try:
        result = subprocess.run(
            f'GPAW_NO_C_EXTENSION=1 GPAW_CPUPY=1 '
            f'python {__file__} {dtype} {gpu} read_write',
            shell=True, capture_output=True,
            text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        print(e.stderr)
        raise e
    print(result.stdout)


@pytest.mark.serial
@pytest.mark.parametrize('dtype',
                         ['np.complex128',
                          'np.complex64',
                          'np.float64',
                          'np.float32'])
@pytest.mark.parametrize('gpu', [False, True])
def test_restart_fixed_density(in_tmp_dir, dtype, gpu):
    try:
        result = subprocess.run(
            f'GPAW_NO_C_EXTENSION=1 GPAW_CPUPY=1 '
            f'python {__file__} {dtype} {gpu} restart_fixed',
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
def test_read_write_gpw_gpu(in_tmp_dir, dtype):
    run_read_write_gpw(dtype=dtype, gpu='True')


@pytest.mark.serial
@pytest.mark.gpu
@pytest.mark.skipif(cupy_is_fake, reason='No cupy')
@pytest.mark.parametrize('dtype',
                         [np.complex128,
                          np.complex64,
                          np.float64,
                          np.float32])
def test_restart_fixed_density_gpu(in_tmp_dir, dtype):
    run_restart_fixed_density(dtype=dtype, gpu='True')


def run_read_write_gpw(dtype, gpu):
    """Test basic GPW file write and read operations"""
    atoms = molecule('H2O')
    atoms.center(vacuum=2.5)

    gpu = gpu == 'True'
    gpw_file = 'test.gpw'

    # Create initial calculation and write GPW file
    atoms.calc = GPAW(xc={'name': 'LDA'},
                      symmetry='off',
                      random=True,
                      convergence={'energy': 1e-5,
                                   'forces': 1e-3,
                                   'eigenstates': 1e-6},
                      mode={'name': 'pw',
                            'ecut': 200.0,
                            'dtype': dtype},
                      parallel={'gpu': gpu},
                      txt=None)

    e_pot_original = atoms.get_potential_energy()
    expected_e = 9.595593472328737

    assert atoms.calc.wfs.dtype == dtype
    assert e_pot_original == pytest.approx(expected_e, rel=1e-3)

    # Write with explicit precision to match the dtype
    if dtype in [np.float32, np.complex64]:
        precision = 'single'
        expected_read_dtype = dtype
    else:
        precision = 'double'
        expected_read_dtype = dtype

    atoms.calc.write(gpw_file, precision=precision)

    # Test basic read from GPW file
    calc_restart = GPAW(gpw_file)
    e_pot_restart = calc_restart.get_potential_energy()

    assert calc_restart.wfs.dtype == expected_read_dtype
    assert e_pot_restart == pytest.approx(expected_e, rel=1e-3)


def run_restart_fixed_density(dtype, gpu):
    """
    Test restart from GPW file and fixed density calculation
    with dtype conversions (calc_dtype, write_precision, restart_dtype):
    - float32 -> single -> float32 (control)
    - float64 -> single -> float64
    - complex128 -> single -> complex128
    - float32 -> double -> complex64
    """
    atoms = molecule('H2O')
    atoms.center(vacuum=2.5)

    gpu = gpu == 'True'

    # Define conversion tuples: (calc_dtype, write_precision, restart_dtype)
    conversion_tuples = {
        np.float32: (np.float32, 'single', np.float32),
        np.float64: (np.float64, 'single', np.float64),
        np.complex64: (np.float32, 'double', np.complex64),
        np.complex128: (np.complex128, 'single', np.complex128),
    }

    calc_dtype, write_precision, restart_dtype = conversion_tuples[dtype]

    print(f"Testing: {calc_dtype} -> {write_precision} -> {restart_dtype}")
    gpw_file = 'test.gpw'
    atoms.calc = GPAW(xc={'name': 'LDA'},
                      symmetry='off',
                      random=True,
                      convergence={'energy': 1e-5, 'forces': 1e-3,
                                   'eigenstates': 1e-6},
                      mode={'name': 'pw',
                            'ecut': 200.0,
                            'dtype': calc_dtype},
                      parallel={'gpu': gpu}, txt=None)

    atoms.get_potential_energy()
    atoms.calc.write(gpw_file, mode='all', precision=write_precision)
    assert atoms.calc.wfs.dtype == calc_dtype

    # When reading from GPW file, restore original calculation's dtype
    # The stored precision is just for storage efficiency
    expected_dtype = calc_dtype
    calc_read = GPAW(gpw_file, txt=None)

    assert calc_read.wfs.dtype == expected_dtype

    fixed_dens_calc = calc_read.fixed_density(
        mode={'name': 'pw', 'ecut': 201.0, 'dtype': restart_dtype},
        parallel={'gpu': gpu}, txt=None)
    e_pot_fixed = fixed_dens_calc.get_potential_energy()
    expected_e = 9.595593472328737

    assert fixed_dens_calc.wfs.dtype == restart_dtype
    assert e_pot_fixed == pytest.approx(expected_e, rel=1e-3)


if __name__ == '__main__':
    dtypes = {'np.float32': np.float32,
              'np.float64': np.float64,
              'np.complex64': np.complex64,
              'np.complex128': np.complex128}

    dtype_str = sys.argv[1]
    gpu_str = sys.argv[2]
    test_type = sys.argv[3]

    if test_type == 'read_write':
        run_read_write_gpw(dtype=dtypes[dtype_str], gpu=gpu_str)
    elif test_type == 'restart_fixed':
        run_restart_fixed_density(dtype=dtypes[dtype_str], gpu=gpu_str)
