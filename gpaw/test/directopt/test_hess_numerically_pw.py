import pytest

from gpaw import GPAW
from gpaw.directmin.derivatives import Derivatives
from gpaw.directmin.etdm_fdpw import FDPWETDM
from gpaw.mom import prepare_mom_calculation


@pytest.mark.do
def test_hess_numerically_pw(in_tmp_dir, gpw_files):
    """
    Test complex numerical Hessian
    w.r.t rotation parameters in LCAO

    :param in_tmp_dir:
    :return:
    """

    calc = GPAW(gpw_files['h_hess_num_pw'])
    atoms = calc.atoms
    atoms.calc = calc

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
