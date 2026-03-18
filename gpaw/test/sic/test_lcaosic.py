import io

import numpy as np
import numpy.testing as npt
import pytest

from gpaw import GPAW
from gpaw.mpi import world
from gpaw.old.logger import GPAWLogger
from gpaw.old.wavefunctions.base import eigenvalue_string
from gpaw.test.sic._utils import (MockWorld, extract_lagrange_section,
                                  mk_arr_from_str)


@pytest.mark.old_gpaw_only
@pytest.mark.sic
def test_lcaosic_innerloop(in_tmp_dir, gpw_files):
    """Test that inner loop PZ localization gives same energy as without."""
    calc_no_inner = GPAW(gpw_files['h2o_lcaosic'])
    atoms_no_inner = calc_no_inner.atoms
    atoms_no_inner.calc = calc_no_inner
    e_no_inner = atoms_no_inner.get_potential_energy()

    calc_inner = GPAW(gpw_files['h2o_lcaosic_innerloop'])
    atoms_inner = calc_inner.atoms
    atoms_inner.calc = calc_inner
    e_inner = atoms_inner.get_potential_energy()

    assert e_no_inner == pytest.approx(e_inner, abs=1e-3)


@pytest.mark.old_gpaw_only
@pytest.mark.sic
def test_lcaosic(in_tmp_dir, gpw_files):
    """
    Test Perdew-Zunger Self-Interaction
    Correction  in LCAO mode using ETDM
    :param in_tmp_dir:
    :return:
    """

    # Water molecule:
    calc = GPAW(gpw_files['h2o_lcaosic'])
    H2O = calc.atoms
    H2O.calc = calc
    e = H2O.get_potential_energy()
    f = H2O.get_forces()

    assert e == pytest.approx(-12.16352, abs=1e-3)

    f2 = np.array([[-4.21747862, -4.63118948, 0.00303988],
                   [5.66636141, -0.51037693, -0.00049136],
                   [-1.96478031, 5.4043045, -0.0006107]])
    assert f2 == pytest.approx(f, abs=0.1)

    numeric = False
    if numeric:
        from gpaw.test import calculate_numerical_forces
        f_num = calculate_numerical_forces(H2O, 0.001)
        print('Numerical forces')
        print(f_num)
        print(f - f_num, np.abs(f - f_num).max())

    calc.write('h2o.gpw', mode='all')
    from gpaw import restart
    H2O, calc = restart('h2o.gpw', txt='-')
    H2O.positions += 1.0e-6
    f3 = H2O.get_forces()
    niter = calc.get_number_of_iterations()
    assert niter == pytest.approx(4, abs=3)
    assert f2 == pytest.approx(f3, abs=0.1)

    if world.rank == 0:
        logger = GPAWLogger(MockWorld(rank=0))
        string_io = io.StringIO()
        logger.fd = string_io
        calc.wfs.summary_func(logger)
        lstr = extract_lagrange_section(string_io.getvalue())

        expect_lagrange_str = """\
          Band         L_ii  Occupancy
             0    -20.96885    2.00000
             1    -20.72880    2.00000
             2    -14.63714    2.00000
             3    -14.63436    2.00000
             4      1.52758    0.00000
             5      5.15451    0.00000
        """
        expect_eigen_str = """\
         Band  Eigenvalues  Occupancy
              0    -30.11715    2.00000
              1    -17.20818    2.00000
              2    -12.38599    2.00000
              3    -11.25782    2.00000
              4      1.52757    0.00000
              5      5.15452    0.00000
        """

        npt.assert_allclose(
            mk_arr_from_str(expect_lagrange_str),
            mk_arr_from_str(lstr),
            atol=0.3,
        )

        npt.assert_allclose(
            mk_arr_from_str(expect_eigen_str),
            mk_arr_from_str(eigenvalue_string(calc.wfs)),
            atol=0.3,
        )
