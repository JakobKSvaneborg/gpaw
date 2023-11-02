import pytest
import numpy as np

from gpaw import GPAW, PW
from gpaw.directmin.etdm_fdpw import FDPWETDM
from gpaw.directmin.derivatives import Derivatives
from gpaw.mom import prepare_mom_calculation
from ase import Atoms


@pytest.mark.do
def test_gradient_numerically_pw(in_tmp_dir):
    """
    Test analytical vs. numerical gradients exponential
    transformation in pw
    :param in_tmp_dir:
    :return:
    """

    for complex in [False, True]:
        atoms = Atoms('H3', positions=[(0, 0, 0),
                                       (0.59, 0, 0),
                                       (1.1, 0, 0)])
        atoms.center(vacuum=2.0)
        atoms.set_pbc(True)
        calc = GPAW(mode=PW(300, force_complex_dtype=complex),
                    basis='sz(dzp)',
                    h=0.3,
                    spinpol=False,
                    convergence={'energy': np.inf,
                                 'eigenstates': np.inf,
                                 'density': np.inf,
                                 'minimum iterations': 1},
                    eigensolver=FDPWETDM(converge_unocc=True),
                    occupations={'name': 'fixed-uniform'},
                    mixer={'backend': 'no-mixing'},
                    nbands='nao',
                    symmetry='off',
                    )
        atoms.calc = calc
        atoms.get_potential_energy()

        calc.set(eigensolver=FDPWETDM(excited_state=True))
        f_sn = [calc.get_occupation_numbers(spin=s).copy() / 2
                for s in range(calc.wfs.nspins)]
        prepare_mom_calculation(calc, atoms, f_sn, use_fixed_occupations=True)
        atoms.get_potential_energy()

        ham = calc.hamiltonian
        wfs = calc.wfs
        dens = calc.density
        der = Derivatives(wfs.eigensolver.outer_iloop, wfs,
                          update_c_ref=True, random_amat=True)

        g_a = der.get_analytical_derivatives(
            wfs.eigensolver.outer_iloop, ham, wfs, dens)
        g_n = der.get_numerical_derivatives(
            wfs.eigensolver.outer_iloop, ham, wfs, dens)

        iut = np.triu_indices(der.a[0].shape[0], 1)
        assert g_n[0].real == pytest.approx(g_a[0][iut].real, abs=1.0e-4)
        assert g_n[0].imag == pytest.approx(g_a[0][iut].imag, abs=1.0e-4)
