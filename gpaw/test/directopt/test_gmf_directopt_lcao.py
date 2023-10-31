import pytest

from gpaw import GPAW, LCAO
from gpaw.directmin.tools import excite
from gpaw.directmin.etdm_lcao import LCAOETDM

from ase import Atoms
import numpy as np


@pytest.mark.do
def test_gmf_directopt_lcao(in_tmp_dir):
    # Water molecule:
    d = 0.9575
    t = np.pi / 180 * 104.51
    H2O = Atoms('OH2',
                positions=[(0, 0, 0),
                           (d, 0, 0),
                           (d * np.cos(t), d * np.sin(t), 0)])
    H2O.center(vacuum=4.0)

    calc = GPAW(mode=LCAO(),
                basis='dzp',
                h=0.22,
                occupations={'name': 'fixed-uniform'},
                eigensolver='etdm-lcao',
                mixer={'backend': 'no-mixing'},
                nbands='nao',
                symmetry='off',
                spinpol=True,
                convergence={'density': 1.0e-4,
                             'eigenstates': 4.0e-8})
    H2O.calc = calc
    H2O.get_potential_energy()

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
