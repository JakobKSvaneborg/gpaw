import numpy as np
import pytest
from ase.io.ulm import ulmopen
from ase.parallel import parprint
from ase import Atom, Atoms

from gpaw import GPAW


@pytest.mark.mgga
def test_pw_forces_mgga(gpaw_new):
    a = 4.05
    d = a / 2**0.5
    bulk = Atoms([Atom('Al', (0, 0, 0)),
                  Atom('Al', (0.49, 0.48, 0.47))], pbc=True)
    bulk.set_cell((d, d, a), scale_atoms=True)
    bulk.calc = GPAW(mode='pw',
                     xc='revTPSS',
                     nbands=2 * 8,
                     kpts=(2, 2, 2))

    f_test = bulk.get_forces()
    parprint('Forces:\n', f_test)
    assert pytest.approx(f_test) != [0., 0., 0.]
    # Pre-calculated forces:
    f_revTPSS = np.array([[-0.10154195, -0.20316854, -0.17164669],
                          [ 0.10132630,  0.20300123,  0.17154444]])

    f_err = f_test - f_revTPSS

    # New GPAW and old GPAW disagree by ca. ± 7e-4 so I'll set treshold to 1e-3
    assert np.all(abs(f_err) < 1e-3)
    parprint('Error in forces:\n', f_err)
