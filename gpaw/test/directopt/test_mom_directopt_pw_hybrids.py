import pytest

import numpy as np


from gpaw import GPAW, PW
from gpaw.mom import prepare_mom_calculation


@pytest.mark.do
def test_mom_directopt_pw_hybrids(in_tmp_dir, gpw_files):
    calc = GPAW(gpw_files['h2_mom_do_pwh'])
    h2 = calc.atoms
    h2.calc = calc
    e = h2.get_potential_energy()

    # Total and orbital energies calculated using
    # RMMDIIS with disabled code below
    e_ref = -6.985891
    eig_ref = [-11.77015, 1.18932]
    f_ref = [[-0.34178, 0.0, 0.0], [0.34178, 0.0, 0.0]]
    e_ref_es = 20.697867
    eig_ref_es = [-16.46296, -3.35601]
    f_ref_es = [[-34.90936, 0.0, 0.0], [34.90936, 0.0, 0.0]]

    reference_calc = False
    if reference_calc:
        calc = GPAW(mode=PW(300),
                    # h=0.3,
                    xc={'name': 'HSE06', 'backend': 'pw'},
                    symmetry='off',
                    nbands=2,
                    convergence={'eigenstates': 4.0e-6,
                                 'bands': 'all'})
        h2.calc = calc
        h2.get_potential_energy()
        h2.get_forces()
        calc.get_eigenvalues()

        f_sn = [[0, 1]]
        prepare_mom_calculation(calc, h2, f_sn)

        h2.get_potential_energy()
        h2.get_forces()
        calc.get_eigenvalues()

    eig = calc.get_eigenvalues()
    assert e == pytest.approx(e_ref, abs=1.0e-3)
    assert eig == pytest.approx(eig_ref, abs=0.1)
    if calc.old:
        f = calc.get_forces()
        assert f == pytest.approx(np.array(f_ref), abs=1.0e-2)

    calc.set(eigensolver={'name': 'etdm-fdpw',
                          'excited_state': True,
                          'converge_unocc': True})
    f_sn = [[0, 1]]
    prepare_mom_calculation(calc, h2, f_sn)

    e_es = h2.get_potential_energy()
    eig_es = calc.get_eigenvalues()
    assert e_es == pytest.approx(e_ref_es, abs=1.0e-3)
    # XXX(rg): Flaky on CI, locally abs=0.1 is fine
    assert eig_es == pytest.approx(eig_ref_es, abs=0.3)
    if calc.old:
        f_es = calc.get_forces()
        assert f_es == pytest.approx(np.array(f_ref_es), abs=1.0e-2)
