import numpy as np
import pytest
from ase import Atoms

from gpaw import GPAW
from gpaw.mpi import world


@pytest.mark.new_gpaw_ready
@pytest.mark.hybrids
def test_pawexxvv():
    from _gpaw import pawexxvv

    from gpaw.hybrids.paw import python_pawexxvv
    for i in range(20):
        D_ii = np.random.rand(i, i)
        p = i * (i + 1) // 2
        M_pp = np.random.rand(p, p)
        V_ii = python_pawexxvv(M_pp, D_ii)
        V2_ii = pawexxvv(M_pp, D_ii)
        assert np.allclose(V_ii, V2_ii)


@pytest.mark.new_gpaw_ready
@pytest.mark.hybrids
# @pytest.mark.parametrize('ccirs', [False, True])
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('eigensolver', ['davidson', 'ppcg'])
def test_hse06(gpaw_new, dtype, eigensolver):

    if not gpaw_new and eigensolver == 'ppcg':
        pytest.skip('PPCG only for GPAW new.')

    atoms = Atoms('Li2', [[0, 0, 0], [0, 0, 2.0]])
    atoms.center(vacuum=2.5)
    atoms.calc = GPAW(
        mode=dict(name='pw',
                  force_complex_dtype=not gpaw_new and dtype is complex),
        xc='HSE06',
        eigensolver=eigensolver,
        convergence={'density': 1e-6},
        parallel={'domain': world.size},
        nbands=4)
    e = atoms.get_potential_energy()
    assert e == pytest.approx(-5.633278, abs=1e-4)
    eigs = atoms.calc.get_eigenvalues(spin=0)
    assert eigs[0] == pytest.approx(-4.67477532, abs=1e-4)
    f = atoms.get_forces()
    f0 = 2.35055
    assert f == pytest.approx(np.array([[0, 0, -f0], [0, 0, f0]]), abs=3e-4)


@pytest.mark.new_gpaw_ready
@pytest.mark.hybrids
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('eigensolver', ['davidson', 'ppcg'])
def test_h(gpaw_new, dtype, eigensolver):

    if not gpaw_new and eigensolver == 'ppcg':
        pytest.skip('PPCG only for GPAW new.')

    atoms = Atoms('H', magmoms=[1])
    atoms.center(vacuum=2.5)
    atoms.calc = GPAW(mode=dict(name='pw',
                                force_complex_dtype=dtype is complex),
                      xc='HSE06',
                      eigensolver=eigensolver,
                      nbands=2,
                      parallel={'kpt': 1, 'band': 1, 'domain': world.size},
                      convergence={'density': 1e-6})
    e = atoms.get_potential_energy()
    eigs = atoms.calc.get_eigenvalues(spin=0)
    assert e == pytest.approx(-1.7041, abs=4e-4)
    assert eigs[0] == pytest.approx(-9.7143, abs=4e-4)


if __name__ == '__main__':
    test_hse06(1, complex, 'ppcg')
