import pytest

import numpy as np

from ase import Atoms
from ase.units import Bohr

from gpaw import GPAW, FD
from gpaw.directmin.etdm_fdpw import FDPWETDM
from gpaw.xc.hybrid import HybridXC
from gpaw.eigensolvers import RMMDIIS
from gpaw.mom import prepare_mom_calculation


@pytest.mark.do
def test_mom_directopt_fd_hybrids(in_tmp_dir):
    d = 1.4 * Bohr
    h2 = Atoms('H2',
               positions=[[-d / 2, 0, 0],
                          [d / 2, 0, 0]])
    h2.center(vacuum=3)

    # Total and orbital energies calculated using
    # RMMDIIS with disabled code below
    e_ref = -5.969348
    eig_ref = np.array([-11.94789695, 2.02930128])
    e_ref_es = 21.861924
    eig_ref_es = np.array([-15.98483801, -3.4381461])

    calc = GPAW(mode=FD(),
                h=0.3,
                xc=HybridXC('PBE0', unocc=True),
                eigensolver=FDPWETDM(converge_unocc=True),
                mixer={'backend': 'no-mixing'},
                occupations={'name': 'fixed-uniform'},
                symmetry='off',
                nbands=3,
                convergence={'eigenstates': 4.0e-6},
                )
    h2.calc = calc
    e = h2.get_potential_energy()
    eig = calc.get_eigenvalues()

    calc.set(eigensolver=FDPWETDM(excited_state=True,
                                  converge_unocc=True))
    f_sn = [[0, 1, 0]]
    prepare_mom_calculation(calc, h2, f_sn)

    e_es = h2.get_potential_energy()
    eig_es = calc.get_eigenvalues()

    assert e == pytest.approx(e_ref, abs=1.0e-3)
    assert eig[:-1] == pytest.approx(eig_ref, abs=0.1)
    assert e_es == pytest.approx(e_ref_es, abs=1.0e-3)
    assert eig_es[:-1] == pytest.approx(eig_ref_es, abs=0.1)

    reference_calc = False
    if reference_calc:
        calc = GPAW(mode=FD(),
                    h=0.3,
                    xc=HybridXC('PBE0', unocc=True),
                    symmetry='off',
                    nbands=3,
                    convergence={'eigenstates': 4.0e-6,
                                 'bands': 2},
                    eigensolver=RMMDIIS()
                    )
        h2.calc = calc
        h2.get_potential_energy()
        calc.get_eigenvalues()[:-1]

        f_sn = [[0, 1, 0]]
        prepare_mom_calculation(calc, h2, f_sn)

        h2.get_potential_energy()
        calc.get_eigenvalues()[:-1]
