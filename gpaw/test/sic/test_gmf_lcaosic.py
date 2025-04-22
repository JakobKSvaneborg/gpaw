import pytest

from gpaw import GPAW, LCAO
from gpaw.directmin.etdm_lcao import LCAOETDM
from gpaw.directmin.tools import excite
from gpaw.directmin.derivatives import Davidson
from gpaw.mom import prepare_mom_calculation
from ase import Atoms
import numpy as np


@pytest.mark.old_gpaw_only
@pytest.mark.sic
def test_gmf_lcaosic(in_tmp_dir, gpw_files):
    """
    test Perdew-Zunger Self-Interaction
    Correction  in LCAO mode using DirectMin
    :param in_tmp_dir:
    :return:
    """
    calc = GPAW(gpw_files['h2o_gmf_lcaosic'])
    H2O = calc.atoms
    H2O.calc = calc
    e = H2O.get_potential_energy()
    f = H2O.get_forces()

    f_num = np.array([[-8.01206297e+00, -1.51553367e+01, 3.60670227e-03],
                      [1.42287594e+01, -9.81724693e-01, -5.09333905e-04],
                      [-4.92299436e+00, 1.55306540e+01, 2.12438557e-03]])

    numeric = False
    if numeric:
        from gpaw.test import calculate_numerical_forces
        f_num = calculate_numerical_forces(H2O, 0.001)
        print('Numerical forces')
        print(f_num)
        print(f - f_num, np.abs(f - f_num).max())

    assert e == pytest.approx(-2.007241, abs=1.0e-3)
    assert f == pytest.approx(f_num, abs=0.75)
