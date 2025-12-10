import pytest
from ase import Atoms
from ase.build import bulk

from gpaw import GPAW
from gpaw.mpi import world


@pytest.mark.parametrize('mode', ['pw', 'fd'])
@pytest.mark.parametrize('element', ['Al', 'Si'])
@pytest.mark.parametrize('eigensolver', ['davidson', 'ppcg', 'dir_opt'])
def test_eigensolver(mode, element, eigensolver, gpaw_new):
    if not gpaw_new:
        pytest.skip('Only implemented for new GPAW')

    # element = 'Si'
    # eigensolver = 'davidson'
    # eigensolver = 'ppcg'
    # eigensolver = 'dir_opt'
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
