import pytest
from ase import Atoms
from gpaw import GPAW, PW


@pytest.mark.new_gpaw_ready
@pytest.mark.hybrids
@pytest.mark.parametrize('dtype', [float, complex])
def test_h2(in_tmp_dir, dtype):
    L = 2.6
    a = Atoms('H2',
              [[0, 0, 0], [0.5, 0.5, 0]],
              cell=[L, L, 1],
              pbc=1)
    a.center()

    a.calc = GPAW(
        mode=PW(400, force_complex_dtype=dtype == complex),
        symmetry='off',
        # kpts={'size': (1, 1, 2)},  # 'gamma': not True},
        convergence={'density': 1e-6},
        eigensolver={'name': 'dav', 'niter': 1},
        nbands=1,
        # spinpol=True,
        # txt='H2.txt',
        # setups='ae',
        xc='HSE06')
    e = a.get_potential_energy()
    eigs = a.calc.get_eigenvalues()
    assert e == pytest.approx(-60.161445)
    assert eigs == pytest.approx([-54.15957])
