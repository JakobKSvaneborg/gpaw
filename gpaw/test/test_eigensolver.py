import pytest
from ase import Atoms
from ase.build import bulk, molecule

from gpaw import GPAW
from gpaw.mpi import world


@pytest.mark.parametrize('mode', ['pw'])
@pytest.mark.parametrize('eigensolver', ['ppcg', 'etdm-fdpw'])
@pytest.mark.parametrize('setup', ['paw', 'ae'])
def test_ae(mode, eigensolver, setup, gpaw_new):
    if not gpaw_new:
        pytest.skip('Only implemented for new GPAW')

    occupations = {'name': 'fermi-dirac', 'width': 0.01}
    mixer = {'backend': 'fft'}
    if eigensolver == 'etdm-fdpw':
        eigensolver = {'name': 'etdm-fdpw', 'converge_unocc': True}
        occupations = {'name': 'fixed',
                       'numbers': [[1, 0, 0, 0], [1, 0, 0, 0]]}
        mixer = {'backend': 'no-mixing'}
        energy_tolerance = 5e-4
    else:
        energy_tolerance = 5e-5
    eig_tolerance = 1e-3
    spinpol = False

    if mode == 'pw':
        mode_d = {'name': 'pw', 'ecut': 400, 'force_complex_dtype': True}
    elif mode == 'fd':
        mode_d = {'name': 'fd'}

    atoms = molecule('H2')
    atoms.center(vacuum=2.0)
    atoms.set_pbc(True)
    e0_t = {'ae': {'pw': -30.5129632}, 'paw': {'pw': -6.9729888}}
    nocc = 1
    eig_t = {'ae': {'pw': [-10.239929, -0.147173, 1.869714, 5.100209]},
             'paw': {'pw': [-10.386695, -0.159743, 1.876333, 5.132689]}}

    params = {'mode': mode_d,
              'nbands': -3,
              'kpts': {'size': [1, 1, 1], 'gamma': True},
              'eigensolver': eigensolver,
              'spinpol': spinpol,
              'occupations': occupations,
              'mixer': mixer,
              'setups': setup,
              'convergence': {'eigenstates': 1e-8,
                              'energy': 1e-5,
                              'bands': 'all'}}

    calc = GPAW(**params)
    atoms.calc = calc
    e0 = atoms.get_potential_energy()
    eig = atoms.calc.get_eigenvalues()
    print(eigensolver, eig)
    if 1:
        atoms.calc.diagonalize_full_hamiltonian(nbands=8)
        eig_exact = atoms.calc.get_eigenvalues()
        print('exact', eig_exact)

    eig_test = eig_t[setup][mode]
    assert e0 == pytest.approx(e0_t[setup][mode], abs=energy_tolerance)
    assert eig[:nocc] == pytest.approx(eig_test[:nocc], abs=eig_tolerance)
    assert eig == pytest.approx(eig_test, abs=eig_tolerance)


@pytest.mark.parametrize('mode', ['pw', 'fd'])
@pytest.mark.parametrize('element', ['Al', 'Si'])
@pytest.mark.parametrize('eigensolver', ['davidson', 'ppcg'])
def test_eigensolver(mode, element, eigensolver, gpaw_new):
    if not gpaw_new:
        pytest.skip('Only implemented for new GPAW')

    energy_tolerance = 5e-5
    eig_tolerance = 1e-3
    spinpol = False

    # enforce band parallelization
    if world.size > 1:
        parallel = {'band': 2}
    else:
        parallel = {'band': None}

    if mode == 'pw':
        mode_d = {'name': 'pw', 'ecut': 400}
    elif mode == 'fd':
        mode_d = {'name': 'fd'}

    if element == 'Si':
        a = 5.431
        atoms = bulk('Si', 'diamond', a=a)
        e0_t = {'pw': -11.7125397, 'fd': -11.7032261}
        # occupied eigenvalues
        nocc = 4
        eig_t = [-4.08139256, -1.23190614, 1.59863588, 2.95648716]
    elif element == 'Al':
        a = 4.05
        d = a / 2**0.5
        atoms = Atoms('Al2', positions=[[0, 0, 0], [.5, .5, .5]], pbc=True)
        atoms.set_cell((d, d, a), scale_atoms=True)
        e0_t = {'pw': -6.9786673, 'fd': -6.9797518}
        nocc = 3
        eig_t = [-1.36400629, 2.97388703, 6.63549518]

    params = {'mode': mode_d,
              'nbands': 2 * 8,
              'kpts': {'size': [2, 2, 2]},
              'eigensolver': eigensolver,
              'spinpol': spinpol,
              'parallel': parallel,
              'convergence': {'eigenstates': 1e-8,
                              'energy': 1e-5}}

    calc = GPAW(**params)
    atoms.calc = calc
    e0 = atoms.get_potential_energy()
    eig = atoms.calc.get_eigenvalues()[:nocc]

    assert e0 == pytest.approx(e0_t[mode], abs=energy_tolerance)
    assert eig == pytest.approx(eig_t, abs=eig_tolerance)
