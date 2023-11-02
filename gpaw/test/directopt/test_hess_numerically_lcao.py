import pytest

from ase import Atoms
from gpaw import GPAW, LCAO
from gpaw.directmin.derivatives import Derivatives
import numpy as np


@pytest.mark.do
def test_hess_numerically_lcao(in_tmp_dir):
    """
    Test complex numerical Hessian
    w.r.t rotation parameters in LCAO

    :param in_tmp_dir:
    :return:
    """

    calc = GPAW(xc='PBE',
                mode=LCAO(force_complex_dtype=True),
                h=0.25,
                basis='dz(dzp)',
                spinpol=False,
                eigensolver={'name': 'etdm-lcao',
                             'representation': 'u-invar'},
                occupations={'name': 'fixed-uniform'},
                mixer={'backend': 'no-mixing'},
                nbands='nao',
                symmetry='off',
                )

    atoms = Atoms('H', positions=[[0, 0, 0]])
    atoms.center(vacuum=5.0)
    atoms.set_pbc(False)
    atoms.calc = calc
    atoms.get_potential_energy()

    numder = Derivatives(calc.wfs.eigensolver, calc.wfs)

    hess_n = numder.get_numerical_derivatives(
        calc.wfs.eigensolver,
        calc.hamiltonian,
        calc.wfs,
        calc.density,
        what2calc='hessian'
    )
    hess_a = numder.get_analytical_derivatives(
        calc.wfs.eigensolver,
        calc.hamiltonian,
        calc.wfs,
        calc.density,
        what2calc='hessian'
    )
    hess_nt = np.asarray([[1.32720630e+00, -1.93947467e-11],
                         [3.95786680e-09, 1.14599176e+00]])
    assert hess_n == pytest.approx(hess_nt, abs=1e-4)
    assert hess_a == pytest.approx(hess_nt, abs=0.2)

    a_mat_u = {0: [np.sqrt(2) * np.pi / 4.0 + 1.0j * np.sqrt(2) * np.pi / 4.0]}
    c_nm_ref = calc.wfs.eigensolver.dm_helper.reference_orbitals
    calc.wfs.eigensolver.rotate_wavefunctions(calc.wfs, a_mat_u, c_nm_ref)
    calc.wfs.eigensolver.update_ks_energy(
        calc.hamiltonian, calc.wfs, calc.density)
    calc.wfs.eigensolver.get_canonical_representation(
        calc.hamiltonian, calc.wfs, calc.density, sort_eigenvalues=True)
    c_nm = {x: calc.wfs.kpt_u[x].C_nM.copy()
            for x in range(len(calc.wfs.kpt_u))}

    numder = Derivatives(calc.wfs.eigensolver, calc.wfs, c_ref=c_nm)

    hess_n = numder.get_numerical_derivatives(
        calc.wfs.eigensolver,
        calc.hamiltonian,
        calc.wfs,
        calc.density,
        what2calc='hessian'
    )

    hess_a = numder.get_analytical_derivatives(
        calc.wfs.eigensolver,
        calc.hamiltonian,
        calc.wfs,
        calc.density,
        what2calc='hessian'
    )
    hess_nt = np.asarray([[-1.08209601e+00, -1.11022302e-09],
                         [8.50014503e-10, -9.37664521e-01]])
    assert hess_n == pytest.approx(hess_nt, abs=1e-4)
    assert hess_a == pytest.approx(hess_nt, abs=0.2)
