import pytest
from ase import Atoms
from ase.units import Ha

from gpaw.new.calculation import DFTCalculation
from gpaw.mpi import size


@pytest.mark.gpu
@pytest.mark.serial
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('gpu', [False, True])
def test_gpu_pw(dtype, gpu):
    if gpu and dtype == float:
        pytest.skip('P_ani * dH_aii kernel not implemented for float')
    atoms = Atoms('H2')
    atoms.positions[1, 0] = 0.75
    atoms.center(vacuum=1.0)
    dft = DFTCalculation.from_parameters(
        atoms,
        dict(mode={'name': 'pw'},
             dtype=dtype,
             parallel={'gpu': gpu},
             setups='paw'),
        log='-')
    dft.converge()
    dft.energies()
    energy = dft.results['energy'] * Ha
    assert energy == pytest.approx(-16.032945, abs=1e-6)


@pytest.mark.gpu
@pytest.mark.skipif(size > 2, reason='Not implemented')
@pytest.mark.parametrize('gpu', [False, True])
@pytest.mark.parametrize('par', ['domain', 'kpt', 'band'])
def test_gpu_pw_k(gpu, par):
    atoms = Atoms('H', pbc=True, cell=[1.0, 1.1, 1.1])
    dft = DFTCalculation.from_parameters(
        atoms,
        dict(mode={'name': 'pw'},
             spinpol=True,
             kpts=(4, 1, 1),
             parallel={'gpu': gpu,
                       par: size},
             setups='paw'),
        log='-')
    dft.converge()
    dft.energies()
    dft.forces()
    dft.stress()
    energy = dft.results['energy'] * Ha
    assert energy == pytest.approx(-17.653433, abs=1e-6)
