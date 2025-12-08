import pytest
from ase import Atom, Atoms

from gpaw import GPAW, RMMDIIS, Mixer


@pytest.mark.parametrize('mode', ['pw', 'fd'])
def test_eigen_blocked_rmm_diis(in_tmp_dir, gpaw_new, mode):
    a = 4.0
    n = 20
    d = 1.0
    x = d / 3**0.5
    atoms = Atoms([Atom('C', (0.0, 0.0, 0.0)),
                   Atom('H', (x, x, x)),
                   Atom('H', (-x, -x, x)),
                   Atom('H', (x, -x, -x)),
                   Atom('H', (-x, x, -x))],
                  cell=(a, a, a), pbc=True)
    base_params = dict(
        mode=mode,
        gpts=(n, n, n),
        nbands=4,
        mixer=Mixer(0.25, 3, 1))
    if gpaw_new:
        es = {'name': 'rmm-diis',
              'niter': 1,
              'diis_steps': 2,
              'trial_step': 0.1}
    else:
        es = {'name': 'rmm-diis',
              'blocksize': 3,
              'niter': 3}
    calc = GPAW(**base_params,
                txt='a.txt',
                eigensolver=es)
    atoms.calc = calc
    e0 = atoms.get_potential_energy()
    niter0 = calc.get_number_of_iterations()

    if gpaw_new:
        # set max_buffer_mem to 0 to ensure blocksize of 1
        es['max_buffer_mem'] = 0
    else:
        es['blocksize'] = 3
    calc = GPAW(**base_params,
                txt='b.txt',
                eigensolver=es)
    atoms.calc = calc
    e1 = atoms.get_potential_energy()
    niter1 = calc.get_number_of_iterations()
    assert e0 == pytest.approx(e1, abs=0.000001)
    assert niter0 == pytest.approx(niter1, abs=0) == 19 if mode == 'fd' else 14
