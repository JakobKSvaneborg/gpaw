import pytest
from ase.build import bulk

from gpaw import GPAW

@pytest.mark.parametrize('eigensolver', ['davidson', 'ppcg', 'dir_opt'])
def test_eigensolver(eigensolver, gpaw_new):
    if not gpaw_new:
        pytest.skip('PPCG only implemented for new GPAW')

    #eigensolver = 'davidson'
    #eigensolver = 'ppcg'
    #eigensolver = 'dir_opt'
    energy_tolerance = 5e-5
    eig_tolerance = 1e-3
    e0_t = -11.71253979
    # occupied eigenvalues
    nocc = 4
    eig_t = [-4.08139256, -1.23190614, 1.59863588, 2.95648716]

    a = 5.431
    atoms = bulk('Si', 'diamond', a=a)

    params = {'mode': {'name': 'pw', 'ecut': 400},
              'nbands': 2 * 8, 'kpts': (2, 2, 2),
              'spinpol': True,
              'convergence': {'eigenstates': 1e-8,
                              'energy': 1e-5}}

    calc = GPAW(**params)
    atoms.calc = calc
    e0 = atoms.get_potential_energy()
    eig = atoms.calc.get_eigenvalues()[:nocc]
    assert e0 == pytest.approx(e0_t, abs=energy_tolerance)
    assert eig == pytest.approx(eig_t, abs=eig_tolerance)
