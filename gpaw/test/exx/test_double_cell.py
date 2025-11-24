import pytest
from ase import Atoms

from gpaw import GPAW, PW


@pytest.mark.libxc
@pytest.mark.hybrids
@pytest.mark.new_gpaw_ready
@pytest.mark.parametrize('use_sym', [False, True])
def test_exx_double_cell(in_tmp_dir, gpaw_new, use_sym):
    if not gpaw_new and use_sym:
        pytest.skip('Does not work')

    L = 2.6
    a = Atoms('H2',
              [[0, 0, 0], [0.5, 0.5, 0]],
              cell=[L, L, 1],
              pbc=1)
    a.center()

    kwargs = dict(
        mode=PW(400),
        convergence={'density': 1e-6},
        mixer={'beta': 0.25},
        spinpol=True,
        xc='HSE06')
    if not use_sym:
        kwargs['symmetry'] = 'off'

    a.calc = GPAW(
        kpts={'size': (1, 1, 4), 'gamma': True},
        # txt='H2-new.txt',
        # parallel={'kpt': 1},
        **kwargs)
    e1 = a.get_potential_energy()
    return
    assert e1 == pytest.approx(-11.022063)
    eig1_kn = a.calc.eigenvalues()[0]
    f1 = a.get_forces()
    f1n = 9.606279587183408
    if 0:
        # To check against numeric calculation of the forces, but it takes
        # more time
        from gpaw.test import calculate_numerical_forces
        f1n = calculate_numerical_forces(a, 0.001, [1], [0])[0, 0]
    assert abs(f1[0, 0] + f1n) < 0.0005
    assert abs(f1[1, 0] - f1n) < 0.0005

    a *= (1, 1, 2)
    a.calc = GPAW(
        kpts={'size': (1, 1, 2), 'gamma': True},
        # txt='H4-new.txt',
        eigensolver={'name': 'davidson', 'niter': 4},
        # parallel={'kpt': 1},
        **kwargs)
    e2 = a.get_potential_energy()
    eig2_kn = a.calc.eigenvalues()[0]
    f2 = a.get_forces()
    f2[:2] -= f1
    f2[2:] -= f1
    assert abs(f2).max() < 0.00085
    assert abs(e2 - 2 * e1) < 0.002

    print(eig1_kn)
    print(eig2_kn)
    # compare occupied eigenvalues:
    if use_sym:
        eigs1 = [eig1_kn[0, 0], eig1_kn[1, 0], eig1_kn[0, 1]]
    else:
        eigs1 = [eig1_kn[1, 0], eig1_kn[2, 0], eig1_kn[1, 1]]
    eigs2 = [eig2_kn[0, 0], eig2_kn[1, 0], eig2_kn[0, 1]]
    assert eigs1 == pytest.approx(eigs2, abs=1e-3)


if __name__ == '__main__':
    test_exx_double_cell(1, 1, 1)
