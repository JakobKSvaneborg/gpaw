import pytest
import numpy as np

from gpaw import GPAW
from gpaw.directmin.etdm_fdpw import FDPWETDM
from gpaw.directmin.derivatives import Derivatives
from gpaw.mom import prepare_mom_calculation


@pytest.mark.do
def test_gradient_numerically_pw(in_tmp_dir, gpw_files):
    """
    Test analytical vs. numerical gradients exponential
    transformation in pw
    :param in_tmp_dir:
    :return:
    """

    tol_between_methods = dict(abs=1.0e-4)
    tol_between_rngs = dict(abs=1.0e-4)

    for calc in [
        GPAW(gpw_files["h3_do_num_pw_complex"]),
        GPAW(gpw_files["h3_do_num_pw"]),
    ]:
        atoms = calc.atoms
        atoms.calc = calc
        # Repeated for False
        atoms.get_potential_energy()

        calc.set(eigensolver=FDPWETDM(excited_state=True))
        f_sn = [calc.get_occupation_numbers(spin=s).copy() / 2
                for s in range(calc.wfs.nspins)]
        prepare_mom_calculation(calc, atoms, f_sn, use_fixed_occupations=True)
        atoms.get_potential_energy()

        ham = calc.hamiltonian
        wfs = calc.wfs
        dens = calc.density

        rngs = [np.random.default_rng(8),
                np.random.default_rng(123456)]
        ders = [Derivatives(wfs.eigensolver.outer_iloop,
                            wfs,
                            random_amat=rng,
                            update_c_ref=True)
                for rng in rngs]

        iut = np.triu_indices(ders[0].a[0].shape[0], 1)
        analytical_results = [
            der.get_analytical_derivatives(
                wfs.eigensolver.outer_iloop, ham, wfs, dens)[0][iut]
            for der in ders]
        numerical_results = [
            der.get_numerical_derivatives(
                wfs.eigensolver.outer_iloop, ham, wfs, dens)[0]
            for der in ders]

        # Test 1: consistency between methods (numerical v. analytic)
        for an, num in zip(analytical_results, numerical_results):
            assert num.real == pytest.approx(an.real, **tol_between_methods)
            assert num.imag == pytest.approx(an.imag, **tol_between_methods)

        # Test 2: consistency between results obtained from different random
        # RNGs
        x, *ys = analytical_results
        for y in ys:
            assert y.real == pytest.approx(x.real, **tol_between_rngs)
            assert y.imag == pytest.approx(x.imag, **tol_between_rngs)
