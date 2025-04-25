import pytest
from ase import Atoms
from gpaw import GPAW
from gpaw.mpi import size
import numpy as np


def test_pawexxvv():
    from gpaw.hybrids.paw import python_pawexxvv
    from _gpaw import pawexxvv
    for i in range(20):
        D_ii = np.random.rand(i, i)
        p = i * (i + 1) // 2
        M_pp = np.random.rand(p, p)
        V_ii = python_pawexxvv(M_pp, D_ii)
        V2_ii = pawexxvv(M_pp, D_ii)
        assert np.allclose(V_ii, V2_ii)


@pytest.mark.parametrize('ccirs', [False, True])
def test_hse06(gpaw_new, ccirs):
    if gpaw_new and size > 4:
        pytest.skip('Only band-parallelization!')
    if gpaw_new:
        experimental = {'ccirs': ccirs}
    else:
        experimental = {}
        if ccirs:
            pytest.skip('CCIRS only for new GPAW')
    atoms = Atoms('Li2', [[0, 0, 0], [0, 0, 2.0]])
    atoms.center(vacuum=2.5)
    # Low max_buffer_mem to test that this value is overwritten due to
    # the non band-local hybrid-xc hamiltonian.
    atoms.calc = GPAW(mode=dict(name='pw', force_complex_dtype=not True),
                      xc='HSE06',
                      experimental=experimental,
                      eigensolver={'name': 'dav', 'max_buffer_mem': 1024 * 4},
                      nbands=4)
    e = atoms.get_potential_energy()
    eigs = atoms.calc.get_eigenvalues(spin=0)
    assert e == pytest.approx(-5.633278, abs=1e-3)
    assert eigs[0] == pytest.approx(-4.67477532, abs=1e-3)


def test_h(gpaw_new):
    if gpaw_new and size > 2:
        pytest.skip('Only band-parallelization!')
    atoms = Atoms('H', magmoms=[1])
    atoms.center(vacuum=2.5)
    atoms.calc = GPAW(mode='pw',
                      xc='HSE06',
                      nbands=2,
                      convergence={'energy': 1e-4})
    e = atoms.get_potential_energy()
    eigs = atoms.calc.get_eigenvalues(spin=0)
    assert e == pytest.approx(-1.703969, abs=4e-3)
    assert eigs[0] == pytest.approx(-9.71440, abs=1e-3)


if __name__ == '__main__':
    test_hse06(1, True)
