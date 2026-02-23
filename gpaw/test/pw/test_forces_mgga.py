import numpy as np
import pytest
from ase.parallel import parprint
from ase import Atom, Atoms

from gpaw import GPAW


@pytest.mark.mgga
def test_pw_forces_mgga(gpaw_new):
    a = 4.05
    d = a / 2**0.5
    bulk = Atoms([Atom('Al', (0, 0, 0)),
                  Atom('Al', (0.48, 0.48, 0.48))], pbc=True)
    bulk.set_cell((d, d, a), scale_atoms=True)

    bulk.calc = GPAW(mode={'name': 'pw', 'ecut': 800},
                     xc='revTPSS',
                     convergence={'forces': 1e-5},
                     kpts=(3, 3, 3),
                     symmetry='off')

    f_test = bulk.get_forces()
    parprint('Forces:\n', f_test)
    assert pytest.approx(f_test) != [0., 0., 0.]

    # Pre-calculated forces (ase.calculators.fd)
    f_revTPSS = np.array([[-0.12277113, -0.12294365, -0.13208065],
                          [0.12294729, 0.12294381, 0.13207365]])
    f_err = f_test - f_revTPSS

    assert np.all(abs(f_err) < 1e-2)
    parprint('Error in forces:\n', f_err)
