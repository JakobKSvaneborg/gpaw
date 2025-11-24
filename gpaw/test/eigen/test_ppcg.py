import pytest
from ase import Atoms

from gpaw import GPAW
from gpaw.mpi import world


def test_eigen_ppcg(gpaw_new):
    if not gpaw_new:
        pytest.skip('PPCG only implemented for new GPAW')

    # eigensolver = 'davidson'
    eigensolver = 'ppcg'
    energy_tolerance = 4e-3
    e0_t = -6.9786673

    a = 4.05
    d = a / 2**0.5
    bulk = Atoms('Al2', positions=[[0, 0, 0], [.5, .5, .5]], pbc=True)
    bulk.set_cell((d, d, a), scale_atoms=True)

    base_params = {'mode': {'name': 'pw', 'ecut': 400},
                   'nbands': 2 * 8, 'kpts': (2, 2, 2)}
    base_convergence = {'eigenstates': 7.2e-9, 'energy': 1e-5}

    calc = GPAW(**base_params, convergence=base_convergence)
    bulk.calc = calc
    e0 = bulk.get_potential_energy()
    assert e0 == pytest.approx(e0_t, abs=5.0e-5)

    calc = GPAW(**base_params,
                convergence={**base_convergence, 'bands': 5},
                eigensolver=eigensolver)
    bulk.calc = calc
    e1 = bulk.get_potential_energy()
    assert e0 == pytest.approx(e1, abs=5.0e-5)

    assert e0 == pytest.approx(-6.97626, abs=energy_tolerance)
    assert e1 == pytest.approx(-6.976265, abs=energy_tolerance)

    # band parallelization
    if world.size % 2 == 0:
        calc = GPAW(**base_params,
                    convergence={**base_convergence, 'bands': 5},
                    parallel={'band': 2},
                    eigensolver=eigensolver)
        bulk.calc = calc
        e3 = bulk.get_potential_energy()
        assert e0 == pytest.approx(e3, abs=5.0e-5)
