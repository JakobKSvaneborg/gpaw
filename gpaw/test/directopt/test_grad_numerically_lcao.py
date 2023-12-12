import pytest
import numpy as np

from gpaw import GPAW, LCAO
from gpaw.directmin.derivatives import Derivatives
from ase import Atoms


@pytest.mark.do
def test_gradient_numerically_lcao(in_tmp_dir):
    """
    test exponential transformation
    direct minimization method for KS-DFT in LCAO
    :param in_tmp_dir:
    :return:
    """

    atoms = Atoms('H3', positions=[(0, 0, 0),
                                   (0.59, 0, 0),
                                   (1.1, 0, 0)])
    atoms.center(vacuum=2.0)
    atoms.set_pbc(True)
    calc = GPAW(mode=LCAO(force_complex_dtype=True),
                basis='sz(dzp)',
                h=0.3,
                spinpol=False,
                convergence={'eigenstates': 10.0,
                             'density': 10.0,
                             'energy': 10.0},
                occupations={'name': 'fixed-uniform'},
                eigensolver={'name': 'etdm-lcao',
                             'matrix_exp': 'egdecomp'},
                mixer={'backend': 'no-mixing'},
                nbands='nao',
                symmetry='off',
                txt=None)
    atoms.calc = calc

    params = [{'name': 'etdm-lcao',
               'representation': 'full',
               'matrix_exp': 'egdecomp'},
              {'name': 'etdm-lcao',
               'representation': 'full',
               'matrix_exp': 'pade-approx'},
              {'name': 'etdm-lcao',
               'representation': 'sparse',
               'matrix_exp': 'egdecomp'},
              {'name': 'etdm-lcao',
               'representation': 'sparse',
               'matrix_exp': 'pade-approx'},
              {'name': 'etdm-lcao',
               'representation': 'u-invar',
               'matrix_exp': 'egdecomp'},
              {'name': 'etdm-lcao',
               'representation': 'u-invar',
               'matrix_exp': 'egdecomp-u-invar'}]

    for eigsolver in params:
        print('IN PROGRESS: ', eigsolver)

        calc = calc.new(eigensolver=eigsolver)
        atoms.calc = calc
        if eigsolver['representation'] == 'u-invar':
            with pytest.warns(UserWarning,
                              match="Use representation == 'sparse'"):
                atoms.get_potential_energy()
        else:
            atoms.get_potential_energy()
        ham = calc.hamiltonian
        wfs = calc.wfs
        dens = calc.density

        # Do we get consistent results regardless of randomization?
        rngs = [False,
                np.random.default_rng(123456)]
        num_ders = [Derivatives(wfs.eigensolver,
                                wfs,
                                random_amat=rng,
                                update_c_ref=True)
                    for rng in rngs]

        analytical_results = [
            der.get_analytical_derivatives(
                wfs.eigensolver, ham, wfs, dens)[0]
            for der in num_ders]
        numerical_results = [
            der.get_numerical_derivatives(
                wfs.eigensolver, ham, wfs, dens)[0]
            for der in num_ders]
        for x, *ys in zip(*analytical_results, *numerical_results):
            assert np.array(ys).real == pytest.approx(x.real, abs=1.0e-2)
            assert np.array(ys).imag == pytest.approx(x.imag, abs=1.0e-2)
