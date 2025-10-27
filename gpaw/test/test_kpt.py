import numpy as np
from ase.dft.kpoints import monkhorst_pack
from gpaw.old.kpt_descriptor import KPointDescriptor, to1bz
from gpaw.dft import MonkhorstPack
from ase import Atoms


def test_kpt():
    k = 70
    k_kc = monkhorst_pack((k, k, 1))
    kd = KPointDescriptor(k_kc + (0.5 / k, 0.5 / k, 0))
    assert (kd.N_c == (k, k, 1)).all()
    assert abs(kd.offset_c - (0.5 / k, 0.5 / k, 0)).sum() < 1e-9

    bzk_kc = np.array([[0.5, 0.5, 0],
                       [0.50000000001, 0.5, 0],
                       [0.49999999999, 0.5, 0],
                       [0.55, -0.275, 0]])
    cell_cv = np.array([[1, 0, 0],
                        [-0.5, 3**0.5 / 2, 0],
                        [0, 0, 5]])
    bz1k_kc = to1bz(bzk_kc, cell_cv)
    error_kc = bz1k_kc - np.array([[0.5, -0.5, 0],
                                   [0.50000000001, -0.5, 0],
                                   [0.49999999999, -0.5, 0],
                                   [0.55, -0.275, 0]])
    assert abs(error_kc).max() == 0.0
    assert not KPointDescriptor(np.zeros((1, 3)) + 1e-14).gamma


def test_even():
    atom = Atoms('H', positions=[(0, 0, 0)], pbc=[1, 1, 1])
    atom.center(vacuum=5)

    kpts_even = MonkhorstPack(density=8,
                              gamma=False,
                              even=True).build(atoms=atom)
    a, b, c = kpts_even.size_c
    l, m, n = kpts_even.shift_c
    assert (a == 6) and (b == 6) and (c == 6)
    assert (l == 0) and (m == 0) and (n == 0)

    kpts_odd = MonkhorstPack(density=8,
                             gamma=False,
                             even=False).build(atoms=atom)
    a, b, c = kpts_odd.size_c
    l, m, n = kpts_odd.shift_c
    assert (a == 7) and (b == 7) and (c == 7)
    assert (np.isclose(l, 0.07142857142857142)
            and np.isclose(m, 0.07142857142857142)
            and np.isclose(n, 0.07142857142857142))

    kpts_none = MonkhorstPack(density=8,
                              gamma=False,
                              even=None).build(atoms=atom)
    a, b, c = kpts_none.size_c
    l, m, n = kpts_none.shift_c
    assert (a == 6) and (b == 6) and (c == 6)
    assert (l == 0) and (m == 0) and (n == 0)
