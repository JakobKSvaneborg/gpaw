import numpy as np
from ase import Atoms
from gpaw import GPAW
from gpaw import PW
import pytest


def test_pw_bulk():
    #bulk = Atoms('Li', pbc=True)
    k = 1
    from ase.build import mx2
    atoms = mx2('WSe2', vacuum=4)
    atoms2 = atoms.copy()
    atoms2.positions += [0, 0, 6]
    atoms = atoms + atoms2
    atoms.center(axis=2, vacuum=4)
    bulk = atoms
    
    calc = GPAW(mode=PW(200),
                kpts=(k, k, k),
                eigensolver={'name': 'rmm-diis',
                             'niter': 5,
                             'trial_step': 0.1})

    bulk.calc = calc
    bulk.get_potential_energy()
    return
    e = []
    A = [2.6, 2.65, 2.7, 2.75, 2.8]
    for a in A:
        bulk.set_cell((a, a, a))
        e.append(bulk.get_potential_energy())

    a = np.roots(np.polyder(np.polyfit(A, e, 2), 1))[0]
    print('a =', a)
    assert a == pytest.approx(2.65247379609, abs=0.001)
