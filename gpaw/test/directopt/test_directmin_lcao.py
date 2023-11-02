import pytest

from gpaw import GPAW, LCAO
from ase import Atoms
import numpy as np
from gpaw.directmin.etdm_lcao import LCAOETDM


@pytest.mark.do
def test_directmin_lcao(in_tmp_dir):
    """
    test exponential transformation
    direct minimization method for KS-DFT in LCAO
    :param in_tmp_dir:
    :return:
    """

    # Water molecule:
    d = 0.9575
    t = np.pi / 180 * 104.51
    H2O = Atoms('OH2',
                positions=[(0, 0, 0),
                           (d, 0, 0),
                           (d * np.cos(t), d * np.sin(t), 0)])
    H2O.center(vacuum=5.0)

    calc = GPAW(mode=LCAO(),
                basis='dzp',
                occupations={'name': 'fixed-uniform'},
                eigensolver='etdm',
                mixer={'backend': 'no-mixing'},
                nbands='nao',
                symmetry='off'
                )
    H2O.calc = calc
    e = H2O.get_potential_energy()

    assert e == pytest.approx(-13.643156256566218, abs=1.0e-4)

    f2 = np.array([[-1.11463, -1.23723, 0.0],
                   [1.35791, 0.00827, 0.0],
                   [-0.34423, 1.33207, 0.0]])

    for use_rho in [0, 1]:
        if use_rho:
            for kpt in calc.wfs.kpt_u:
                kpt.rho_MM = calc.wfs.calculate_density_matrix(kpt.f_n,
                                                               kpt.C_nM)
        f = H2O.get_forces()
        H2O.calc.results.pop('forces')

        assert f2 == pytest.approx(f, abs=1e-2)

    calc.write('h2o.gpw', mode='all')
    from gpaw import restart
    H2O, calc = restart('h2o.gpw', txt='-')
    H2O.positions += 1.0e-6
    f3 = H2O.get_forces()
    niter = calc.get_number_of_iterations()

    assert niter == pytest.approx(3, abs=1)
    assert f2 == pytest.approx(f3, abs=1e-2)

    calc.set(eigensolver=LCAOETDM(
        representation='u-invar',
        matrix_exp='egdecomp-u-invar',
        need_init_orbs=False,
        linesearch_algo={'name': 'max-step'}))
    e = H2O.get_potential_energy()
    niter = calc.get_number_of_iterations()
    assert e == pytest.approx(-13.643156256566218, abs=1.0e-4)
    assert niter == pytest.approx(3, abs=1)
