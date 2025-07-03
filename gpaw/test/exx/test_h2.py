import pytest
from ase import Atoms
from gpaw import GPAW, PW


@pytest.mark.new_gpaw_ready
@pytest.mark.hybrids
@pytest.mark.parametrize('dtype', [float, complex])
def test_h2(in_tmp_dir, dtype, n=2):
    L = 2.6
    a = Atoms('H2',
              [[0, 0, 0], [0.5, 0.5, 0]],
              cell=[L, L, 1],
              pbc=1)
    a.center()

    a *= (1, 1, n)

    t = 'f' if dtype == float else 'c'
    a.calc = GPAW(
        mode=PW(400, force_complex_dtype=dtype == complex),
        symmetry='off',
        kpts={'size': (1, 1, 2 // n), 'gamma': True} if n == 1 else (1, 1, 1),
        convergence={'density': 1e-6},
        # eigensolver={'name': 'davidson', 'niter': 1},
        eigensolver={'name': 'rmm-diis', 'niter': 1},
        nbands=n*2,
        # spinpol=True,
        txt=f'H2-{t}R{n}.txt',
        # setups='ae',
        xc='HSE06'
        )
    e = a.get_potential_energy()
    eigs = a.calc.get_eigenvalues()
    print(e / n)
    print(eigs)
    if n == 1:
        eigs2 = a.calc.get_eigenvalues(kpt=1)
        print(eigs2)
    # assert e == pytest.approx(-60.161445)
    # assert eigs == pytest.approx([-54.15957])


if __name__ == '__main__':
    import sys
    n = int(sys.argv[2])
    if sys.argv[1] == 'f':
        test_h2(1, float, n)
    else:
        test_h2(1, complex, n)
