import pytest

from gpaw import GPAW
from gpaw.directmin.tools import excite
from gpaw.directmin.etdm_lcao import LCAOETDM


@pytest.mark.do
def test_gmf_directopt_lcao(in_tmp_dir, gpw_files):
    # Water molecule:
    calc = GPAW(gpw_files['h2o_do_gmf_lcao'])
    H2O = calc.atoms
    H2O.calc = calc

    # Excited state occupation numbers
    f_sn = excite(calc, 0, 0, spin=(0, 0))

    calc.set(eigensolver=LCAOETDM(
        partial_diagonalizer={'name': 'Davidson', 'logfile': None, 'seed': 42},
        linesearch_algo={'name': 'max-step'},
        searchdir_algo={'name': 'LBFGS-P_GMF'},
        need_init_orbs=False),
        occupations={'name': 'mom', 'numbers': f_sn,
                     'use_fixed_occupations': True})

    e = H2O.get_potential_energy()

    assert e == pytest.approx(-4.8545, abs=1.0e-4)
