import pytest

import numpy as np

from ase import Atoms
from ase.units import Bohr

from gpaw import GPAW, PW
from gpaw.directmin.etdm_fdpw import FDPWETDM
from gpaw.mom import prepare_mom_calculation


@pytest.mark.do
def test_mom_directopt_pw_hybrids(in_tmp_dir):
    d = 1.4 * Bohr
    h2 = Atoms('H2',
               positions=[[-d / 2, 0, 0],
                          [d / 2, 0, 0]])
    h2.center(vacuum=3)

    # Total and orbital energies calculated using
    # RMMDIIS with disabled code below
    e_ref = -7.097909
    eig_ref = np.array([-11.76748297, 1.1921477])
    f_ref = np.array([[-2.43859082e-01, -1.36319407e-08, 1.58245560e-12],
                      [2.43859105e-01, -1.36319716e-08, 1.57184345e-12]])
    e_ref_es = 20.589313
    eig_ref_es = np.array([-16.49121644, -3.35591687])
    f_ref_es = np.array([[-3.48230181e+01, -1.18963128e-13, 1.19836905e-13],
                         [3.48227623e+01, 8.02399669e-14, -1.29486607e-13]])

    calc = GPAW(mode=PW(300),
                h=0.3,
                xc={'name': 'HSE06', 'backend': 'pw'},
                eigensolver=FDPWETDM(converge_unocc=True),
                mixer={'backend': 'no-mixing'},
                occupations={'name': 'fixed-uniform'},
                symmetry='off',
                nbands=2,
                convergence={'eigenstates': 4.0e-6},
                )
    h2.calc = calc
    e = h2.get_potential_energy()
    eig = calc.get_eigenvalues()
    f = calc.get_forces()

    calc.set(eigensolver=FDPWETDM(excited_state=True,
                                  converge_unocc=True))
    f_sn = [[0, 1]]
    prepare_mom_calculation(calc, h2, f_sn)

    e_es = h2.get_potential_energy()
    eig_es = calc.get_eigenvalues()
    f_es = calc.get_forces()

    assert e == pytest.approx(e_ref, abs=1.0e-3)
    assert eig == pytest.approx(eig_ref, abs=0.1)
    assert f == pytest.approx(f_ref, abs=1.0e-2)
    assert e_es == pytest.approx(e_ref_es, abs=1.0e-3)
    assert eig_es == pytest.approx(eig_ref_es, abs=0.1)
    assert f_es == pytest.approx(f_ref_es, abs=1.0e-2)

    reference_calc = False
    if reference_calc:
        calc = GPAW(mode=PW(300),
                    h=0.3,
                    xc={'name': 'HSE06', 'backend': 'pw'},
                    symmetry='off',
                    nbands=2,
                    convergence={'eigenstates': 4.0e-6,
                                 'bands': 'all'},
                    )
        h2.calc = calc
        h2.get_potential_energy()
        calc.get_eigenvalues()
        calc.get_forces()

        f_sn = [[0, 1]]
        prepare_mom_calculation(calc, h2, f_sn)

        h2.get_potential_energy()
        calc.get_eigenvalues()
        calc.get_forces()
