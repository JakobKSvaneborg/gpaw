import pytest
import numpy as np
import subprocess
import sys

from ase.build import molecule
from ase.build import bulk
from ase.build import mx2
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter

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
    #atoms = molecule('H2O')
    #atoms = molecule('C60')
    #atoms.center(vacuum=2.5)
    #atoms = atoms.repeat((2, 2, 1))
    atoms = bulk('Cu')
    atoms = atoms.repeat((2, 2, 2))
    atoms.rattle(stdev=0.001, seed=42)

    gpu = gpu == 'True'

    atoms.calc = GPAW(xc={'name': 'LDA'},
                      symmetry='off',
                      random=True,
                      convergence={'energy': 1e-5,
                                   'eigenstates': 1e-6,
                                   'density': 5e-6,
                                   'forces': 5e-4},
                      kpts={'density': 0.5},
                      mode={'name': 'pw',
                            'ecut': 600.0,
                            'dtype': dtype},
                      parallel={'gpu': gpu},
                      #txt=None
                      )
    atoms.get_potential_energy()

    #opt = BFGS(atoms)
    #opt.run(fmax=0.01)
    #print(atoms.positions)
    '''
    expected_pos = np.array([[ 3.48051776e-01, -9.40451113e-03,  3.56270725e-01],
                             [ 2.14860166e+00,  2.16001619e+00, -3.39153472e-03],
                             [ 2.15837209e+00,  3.48725778e-01,  1.79343851e+00],
                             [ 3.95903045e+00,  1.79703312e+00,  2.15678014e+00],
                             [-3.47492930e-01,  1.81178837e+00,  1.45325343e+00],
                             [ 1.45214200e+00,  3.26015142e+00,  1.81577779e+00],
                             [ 1.46156766e+00,  1.44920222e+00,  3.61314170e+00],
                             [ 3.26244743e+00,  3.61808013e+00,  3.25262878e+00]])
    #'''
    #assert atoms.positions == pytest.approx(expected_pos, abs=1e-3)
    
    #e_pot = atoms.get_potential_energy()
    #expected_e = 9.595593485742606
    #expected_e = -2.19724921704334
    #expected_e = -15.047246
    #expected_e = -582.809148
    #expected_e = -1165.754498
    #assert atoms.calc.wfs.dtype == dtype

    #assert e_pot == pytest.approx(expected_e, rel=1e-3), e_pot - expected_e


if __name__ == '__main__':
    dtypes = {'np.float32': np.float32,
              'np.float64': np.float64,
              'np.complex64': np.complex64,
              'np.complex128': np.complex128}
    run_single_precision(dtype=dtypes[sys.argv[1]], gpu=sys.argv[2])
