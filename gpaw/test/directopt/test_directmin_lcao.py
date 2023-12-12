from typing import Callable, Collection, NamedTuple, Tuple, Union

import numpy as np
import pytest
from ase import Atoms

from gpaw import GPAW, LCAO
from gpaw.directmin.etdm_lcao import LCAOETDM
from gpaw.typing import RNG


@pytest.mark.do
def test_directmin_lcao(in_tmp_dir):
    """
    test exponential transformation
    direct minimization method for KS-DFT in LCAO
    :param in_tmp_dir:
    :return:
    """

    class NiterCheck(NamedTuple):
        assertion: str
        func: Callable[[int], bool]

    is_approx_3 = NiterCheck('niter ~ 3',
                             lambda niter: niter == pytest.approx(3, abs=1))
    is_le_12 = NiterCheck('niter <= 12', lambda niter: niter <= 12)

    def check_niter(niter: int, check: NiterCheck) -> None:
        assert check.func(niter), f'failed {check.assertion} (niter = {niter})'

    target_pot_en = -13.643156256566218
    abs_force_tol, abs_en_tol = 1e-2, 1.0e-4
    lcaoetdm_kwargs = dict(representation='u-invar',
                           matrix_exp='egdecomp-u-invar',
                           need_init_orbs=False,
                           linesearch_algo={'name': 'max-step'})
    lcaoetdm_rand_and_niter_checks: Collection[
        Tuple[Union[RNG, None], NiterCheck]
    ] = [(None, is_approx_3),
         (np.random.default_rng(8), is_le_12)]

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
                symmetry='off')
    H2O.calc = calc
    e = H2O.get_potential_energy()

    assert e == pytest.approx(target_pot_en, abs=abs_en_tol)

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

        assert f2 == pytest.approx(f, abs=abs_force_tol)

    calc.write('h2o.gpw', mode='all')
    from gpaw import restart
    H2O, calc = restart('h2o.gpw', txt='-')
    H2O.positions += 1.0e-6
    f3 = H2O.get_forces()
    niter = calc.get_number_of_iterations()

    check_niter(niter, is_approx_3)
    assert f2 == pytest.approx(f3, abs=abs_force_tol)

    # Test for various randomization options
    for randomize, check in lcaoetdm_rand_and_niter_checks:
        calc.set(eigensolver=LCAOETDM(randomizeorbitals=randomize,
                                      **lcaoetdm_kwargs))
        e = H2O.get_potential_energy()
        niter = calc.get_number_of_iterations()
        check_niter(niter, check)
        assert e == pytest.approx(target_pot_en, abs=abs_en_tol)
