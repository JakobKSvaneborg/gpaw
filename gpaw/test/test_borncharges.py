from gpaw import GPAW
from gpaw.borncharges import born_charges_wf
import numpy as np


def test_born_charges_wf(in_tmp_dir, gpw_files):
    gpw_file = gpw_files["mos2_pw_nosym"]
    calc = GPAW(gpw_file, txt=None)
    atoms = calc.get_atoms()

    Z_avv_test = np.array([[[-1.0732e+00, -2.7710e-04, +4.9039e-10],
                            [+1.9594e-04, -1.0803e+00, -4.0036e-10],
                            [-1.0843e-07, +1.1073e-06, -1.1197e-01]],
                           [[+5.3664e-01, +1.3854e-04, -6.1787e-03],
                            [-9.7972e-05, +5.4017e-01, +3.5707e-03],
                            [+1.0937e-03, +2.6839e-04, +5.5989e-02]],
                           [[+5.3664e-01, +1.3855e-04, +6.1787e-03],
                            [-9.7970e-05, +5.4017e-01, -3.5707e-03],
                            [-1.0936e-03, -2.6950e-04, +5.5989e-02]]])

    Z_avv = born_charges_wf(atoms, gpw_file=gpw_file)['Z_avv']

    assert np.allclose(Z_avv, Z_avv_test, atol=1e-4)
