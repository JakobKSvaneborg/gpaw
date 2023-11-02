import pytest

from ase import Atoms
from gpaw import GPAW, PW
from gpaw.directmin.derivatives import Derivatives
from gpaw.directmin.etdm_fdpw import FDPWETDM
from gpaw.mom import prepare_mom_calculation
import numpy as np


@pytest.mark.do
def test_hess_numerically_pw(in_tmp_dir):
    """
    Test complex numerical Hessian
    w.r.t rotation parameters in LCAO

    :param in_tmp_dir:
    :return:
    """

    atoms = Atoms('H', positions=[[0, 0, 0]])
    atoms.center(vacuum=5.0)
    atoms.set_pbc(False)

    calc = GPAW(xc='PBE',
                mode=PW(300, force_complex_dtype=False),
                h=0.25,
                convergence={'energy': np.inf,
                             'eigenstates': np.inf,
                             'density': np.inf,
                             'minimum iterations': 1},
                spinpol=False,
                eigensolver=FDPWETDM(converge_unocc=True),
                occupations={'name': 'fixed-uniform'},
                mixer={'backend': 'no-mixing'},
                nbands=2,
                symmetry='off',
                )
    atoms.calc = calc
    atoms.get_potential_energy()

    calc.set(eigensolver=FDPWETDM(excited_state=True))
    f_sn = [calc.get_occupation_numbers(spin=s).copy() / 2
            for s in range(calc.wfs.nspins)]
    prepare_mom_calculation(calc, atoms, f_sn)
    atoms.get_potential_energy()

    numder = Derivatives(calc.wfs.eigensolver.outer_iloop, calc.wfs)

    hess_n = numder.get_numerical_derivatives(
        calc.wfs.eigensolver.outer_iloop,
        calc.hamiltonian,
        calc.wfs,
        calc.density,
        what2calc='hessian'
    )
    hess_a = numder.get_analytical_derivatives(
        calc.wfs.eigensolver.outer_iloop,
        calc.hamiltonian,
        calc.wfs,
        calc.density,
        what2calc='hessian'
    )

    hess_nt = 0.464586
    assert hess_n[0] == pytest.approx(hess_nt, abs=1e-3)
    assert hess_a == pytest.approx(hess_n[0], abs=0.2)
