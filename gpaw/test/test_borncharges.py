import pytest
import numpy as np
from glob import glob

from ase.build import mx2, molecule
from gpaw import GPAW
from gpaw.borncharges import born_charges_wf


@pytest.mark.parametrize('use_gpw', [False, True])
def test_born_charges_wf(in_tmp_dir, gpw_files, use_gpw, cleanup=True):
    gpw_file = gpw_files["hbn_pw"]
    calc = GPAW(gpw_file, txt=None)
    atoms = calc.get_atoms()

    Z_t = np.array([[[-2.83053644e+00, +8.72216542e-05, -1.24768192e-06],
                     [+3.14174433e-06, -2.83058107e+00, +5.61120199e-07],
                     [-3.44269736e-06, -3.10374132e-06, -3.42142396e-01]],
                    [[+2.83053644e+00, -8.72216542e-05, +1.24768192e-06],
                     [-3.14174433e-06, +2.83058107e+00, -5.61120199e-07],
                     [+3.44269736e-06, +3.10374132e-06, +3.42142396e-01]]])

    if use_gpw:
        # use the restart file
        Z_avv = born_charges_wf(atoms, gpw_file=gpw_file,
                                cleanup=cleanup)['Z_avv']
    else:
        atoms.calc = calc
        Z_avv = born_charges_wf(atoms, cleanup=cleanup)['Z_avv']

    assert np.allclose(Z_avv, Z_t, atol=1e-4)

    if cleanup:
        flist = glob('disp*.gpw')
        assert len(flist) == 0


@pytest.mark.parametrize('atoms', [mx2('MoS2', vacuum=7.5),
                                   molecule('H2', cell=[10, 10, 10])])
def test_born_charges_ionic_wf(in_tmp_dir, atoms):

    atoms.center()

    Z_a = atoms.numbers
    Z_t = np.array([za * np.eye(3) for za in Z_a])

    Z_avv = born_charges_wf(atoms, ionic_only=True)['Z_avv']

    assert np.allclose(Z_avv, Z_t)
