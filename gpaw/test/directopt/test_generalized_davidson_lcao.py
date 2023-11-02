import pytest

from ase import Atoms
from gpaw import GPAW, LCAO
from gpaw.directmin.derivatives import Davidson
import numpy as np


@pytest.mark.do
def test_generalized_davidson_lcao(in_tmp_dir):
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

    dave = Davidson(calc.wfs.eigensolver, None, 'forward', h=1e-7, seed=42)
    dave.run(calc.wfs, calc.hamiltonian, calc.density)

    hess_nt = np.asarray([[1.32720630e+00, -1.93947467e-11],
                         [3.95786680e-09, 1.14599176e+00]])
    assert dave.lambda_e == pytest.approx(np.sort(np.diag(hess_nt)), abs=1e-4)

    a_mat_u = {0: [np.sqrt(2) * np.pi / 4.0 + 1.0j * np.sqrt(2) * np.pi / 4.0]}
    c_nm_ref = calc.wfs.eigensolver.dm_helper.reference_orbitals
    calc.wfs.eigensolver.rotate_wavefunctions(
        calc.wfs, a_mat_u, c_nm_ref)
    calc.wfs.eigensolver.update_ks_energy(calc.hamiltonian,
                                          calc.wfs, calc.density)
    calc.wfs.eigensolver.get_canonical_representation(
        calc.hamiltonian, calc.wfs, calc.density)
    calc.wfs.eigensolver.dm_helper.set_reference_orbitals(
        calc.wfs, {0: calc.wfs.bd.nbands})

    dave = Davidson(calc.wfs.eigensolver, None, 'central', h=1e-7, seed=42)
    dave.run(calc.wfs, calc.hamiltonian, calc.density)

    hess_nt = np.asarray([[-1.08209601e+00, -1.11022302e-09],
                         [8.50014503e-10, -9.37664521e-01]])
    assert dave.lambda_e == pytest.approx(np.sort(np.diag(hess_nt)), abs=1e-4)
