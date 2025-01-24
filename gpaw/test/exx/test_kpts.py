"""Test case where q=k1-k2 has component outside 0<=q<1 range."""
from typing import Tuple

import pytest
import numpy as np
from ase import Atoms

from gpaw import GPAW, PW
from gpaw.hybrids.eigenvalues import non_self_consistent_eigenvalues
from gpaw.mpi import size

n = 7


@pytest.fixture(scope='module')
def atoms() -> Atoms:
    a = Atoms('HH',
              cell=[2, 2, 2.5, 90, 90, 60],
              pbc=1,
              positions=[[0, 0, 0], [0, 0, 0.75]])
    parallel = dict(zip(['domain', 'kpt', 'band'],
                        {1: [1, 1, 1],
                         2: [2, 1, 1],
                         4: [2, 2, 1],
                         8: [2, 2, 2]}[size]))
    a.calc = GPAW(mode=PW(200),
                  kpts=(n, n, 1),
                  xc='PBE',
                  parallel=parallel)
    a.get_potential_energy()
    return a


def bandgap(eps: np.ndarray) -> Tuple[int, int, float]:
    """Find band-gap."""
    k1 = eps[0, :, 0].argmax()
    k2 = eps[0, :, 1].argmin()
    return k1, k2, eps[0, k2, 1] - eps[0, k1, 0]


gaps = {'EXX': 21.45,
        'PBE0': 13.93,
        'HSE06': 14.44,
        'PBE': 11.63}


@pytest.mark.libxc
@pytest.mark.hybrids
@pytest.mark.parametrize('xc', ['EXX', 'PBE0', 'HSE06'])
def test_kpts(xc: str, atoms: Atoms) -> None:
    c = atoms.calc
    e0, v0, v = non_self_consistent_eigenvalues(c, xc)
    e = e0 - v0 + v
    k1, k2, gap = bandgap(e)
    assert k1 == 4 and k2 == 7
    assert gap == pytest.approx(gaps[xc], abs=0.01)
    k1, k2, gap = bandgap(e0)
    assert k1 == 4 and k2 == 7
    assert gap == pytest.approx(gaps['PBE'], abs=0.01)


def test_1ds():
    from gpaw.new.pw.nschse import NonSelfConsistentHSE06, ibz2bz
    from gpaw.new.ase_interface import GPAW
    a = Atoms('H',
              [[0.5, 1.0, 1.0]],
              cell=[1.0, 2.0, 2.0],
              pbc=(1, 0, 0))
    n = 4
    c1 = a.calc = GPAW(mode=PW(200),
                       # setups='ae',
                       kpts=(n, 1, 1),
                       txt=None)
    a.get_potential_energy()
    c2 = a.calc = GPAW(mode=PW(200),
                       # setups='ae',
                       kpts=(n, 1, 1),
                       symmetry='off',
                       txt=None)
    a.get_potential_energy()
    bz = ibz2bz(c1.dft.ibzwfs,
                c1.dft.density.nt_sR.desc.new(dtype=complex),
                c1.setups,
                c1.dft.relpos_ac,
                0)
    for wfs in c2.dft.ibzwfs:
        print(wfs.kpt_c, wfs.P_ani[0][0, [0, 1, 4]])
    for p in bz:
        print(p.psit_nG.desc.kpt, p.P_ani[0][0, [0, 1, 4]])


def test_1d():
    from gpaw.new.pw.nschse import NonSelfConsistentHSE06
    from gpaw.new.ase_interface import GPAW
    a = Atoms('H',
              [[0.5, 1.0, 1.0]],
              cell=[1.0, 2.0, 2.0],
              pbc=(1, 0, 0))
    n = 4
    a.calc = GPAW(mode=PW(400),
                  # setups='ae',
                  kpts=(n, 1, 1),
                  symmetry='off',
                  txt=None)
    a.get_potential_energy()
    #e0, v0, v = non_self_consistent_eigenvalues(a.calc, 'HSE06')
    #e_skn = e0 - v0 + v
    #print(e_skn[0])
    hse = NonSelfConsistentHSE06.from_dft_calculation(a.calc.dft)
    e_kn = hse.calculate(a.calc.dft.ibzwfs)[1]
    print(e_kn)
    #assert e_n == pytest.approx(e_skn[0, k], abs=0.002)


def test_2d():
    from gpaw.new.pw.nschse import NonSelfConsistentHSE06
    from gpaw.new.ase_interface import GPAW
    a = Atoms('Li',
              [[0*0.75, 0*0.75, 1.0]],
              # cell=[1.5, 1.5, 2.0],
              cell=[1.5, 1.5, 2.0, 90, 90, 120],
              pbc=(1, 1, 0))
    n = 4
    a.calc = GPAW(mode=PW(200),
                  # setups='ae',
                  #symmetry='off',
                  kpts=(n, n, 1),
                  txt=None)
    a.get_potential_energy()
    #e0, v0, v = non_self_consistent_eigenvalues(a.calc, 'HSE06')
    #e_skn = e0 - v0 + v
    hse = NonSelfConsistentHSE06.from_dft_calculation(a.calc.dft)
    e0_n, e_n = hse.calculate(a.calc.dft.ibzwfs)
    #print(e0 - e0_n)
    #print(e_n - e_skn[0])
    print(e_n)
    # assert e_n == pytest.approx(e_skn[0, k], abs=0.004)


if __name__ == '__main__':
    test_2d()
