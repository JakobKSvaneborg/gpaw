from math import sqrt

import numpy as np
import pytest

from gpaw.new.brillouin import MonkhorstPackKPoints
from gpaw.new.symmetry import Symmetries, SymmetryAnalysisBug


def test_si():
    """Primitive diamond lattice, with Si lattice parameter."""
    a = 5.475
    cell_cv = .5 * a * np.array([(1, 1, 0), (1, 0, 1), (0, 1, 1)])
    spos_ac = np.array([(.00, .00, .00),
                        (.25, .25, .25)])
    id_a = [1, 1]  # two identical atoms
    pbc_c = np.ones(3, bool)
    mp = MonkhorstPackKPoints((4, 4, 4))

    # Do check
    symm = Symmetries.from_cell(cell_cv, pbc=pbc_c)
    symm = symm.analyze_positions(spos_ac, id_a)
    assert len(symm) == 24
    ibz = mp.reduce(symm, strict=False)
    assert len(ibz) == 10
    a = 3 / 32
    b = 1 / 32
    c = 6 / 32
    assert np.all(ibz.weight_k == [a, b, a, c, c, a, a, a, a, b])
    assert not symm.rotation_scc.sum(0).any()

    # Rotate unit cell and check again:
    cell_cv = a / sqrt(2) * np.array([(1, 0, 0),
                                      (0.5, sqrt(3) / 2, 0),
                                      (0.5, sqrt(3) / 6, sqrt(2.0 / 3))])
    symm = Symmetries.from_cell(cell_cv, pbc=pbc_c)
    symm = symm.analyze_positions(spos_ac, id_a)
    assert len(symm) == 24
    ibz2 = mp.reduce(symm, strict=False)
    assert len(ibz) == 10
    assert abs(ibz.weight_k - ibz2.weight_k).sum() < 1e-14
    assert abs(ibz.kpt_kc - ibz2.kpt_kc).sum() < 1e-14
    assert not symm.rotation_scc.sum(0).any()

    mp = MonkhorstPackKPoints((3, 3, 3))
    ibz = mp.reduce(symm)
    assert len(ibz) == 4
    assert abs(ibz.weight_k * 27 - (1, 12, 6, 8)).sum() < 1e-14


def test_h4():
    # Linear chain of four atoms, with H lattice parameter
    cell_cv = np.diag((8., 5., 5.))
    spos_ac = np.array([[0.125, 0.5, 0.5],
                        [0.375, 0.5, 0.5],
                        [0.625, 0.5, 0.5],
                        [0.875, 0.5, 0.5]])
    id_a = [1, 1, 1, 1]  # four identical atoms
    pbc_c = np.array([1, 0, 0], bool)

    # Do check
    symm = Symmetries.from_cell(cell_cv, pbc=pbc_c)
    symm = symm.analyze_positions(spos_ac, id_a)
    assert len(symm) == 16
    mp = MonkhorstPackKPoints((3, 1, 1))
    ibz = mp.reduce(symm)
    assert len(ibz) == 2
    assert np.all(ibz.weight_k == [1 / 3., 2 / 3.])


def test_2():
    # Rocksalt Ni2O2
    a = 7.92
    x = 2. * np.sqrt(1 / 3)
    y = np.sqrt(1 / 8)
    z1 = np.sqrt(1 / 24)
    z2 = np.sqrt(1 / 6)
    cell_cv = a * np.array([(x, y, -z1), (x, -y, -z1), (x, 0., z2)])
    spos_ac = np.array([[0., 0., 0.],
                        [1. / 2., 1. / 2., 1. / 2.],
                        [1. / 4., 1. / 4., 1. / 4.],
                        [3. / 4., 3. / 4., 3. / 4.]])
    id_a = [1, 2, 3, 3]
    pbc_c = np.array([1, 1, 1], bool)

    # Do check
    symm = Symmetries.from_cell(cell_cv, pbc=pbc_c)
    symm = symm.analyze_positions(spos_ac, id_a)
    assert len(symm) == 12
    mp = MonkhorstPackKPoints((2, 2, 2))
    ibz = mp.reduce(symm)
    assert len(ibz) == 2
    assert np.all(ibz.weight_k == [3 / 4, 1 / 4])


def test_new():
    sym = Symmetries.from_cell([1, 2, 3])
    assert sym.has_inversion
    assert len(sym) == 8
    sym2 = sym.analyze_positions([[0, 0, 0], [0, 0, 0.5]],
                                 ids=[1, 2])
    assert len(sym2) == 8


def test_5x5():
    # This system should have 6 symmetries (identity, two rotations
    # and three mirrors), but our code finds only 4.
    # Following that, k-point reduction of a 5x5 Monkhorst-Pack
    # grid blows up!
    a = 5.6
    with pytest.raises(SymmetryAnalysisBug):
        sym = Symmetries.from_cell_and_atoms(
            [a, a, 9, 90, 90, 60],
            pbc=(1, 1, 0),
            _backwards_compatible=True,
            tolerance=1e-7,
            relative_positions=[[0.33333333, 0.3333333, 0.50058348],
                                [0.66666666, 0.6666666, 0.55294505],
                                [0.0, 0.0, 0.44741016],
                                [0.0, 0.0, 0.68013199],
                                [0.33333333, 0.33333333, 0.31908923],
                                [0.66666667, 0.66666667, 0.64723956],
                                [0.0, 0.0, 0.35260054]],
            ids=[0, 1, 1, 1, 1, 2, 2],
            symmorphic=True)

    sym = Symmetries.from_cell_and_atoms(
        [a, a, 9, 90, 90, 60],
        pbc=(1, 1, 0),
        _backwards_compatible=False,
        tolerance=1e-7,
        relative_positions=[[0.33333333, 0.3333333, 0.50058348],
                            [0.66666666, 0.6666666, 0.55294505],
                            [0.0, 0.0, 0.44741016],
                            [0.0, 0.0, 0.68013199],
                            [0.33333333, 0.33333333, 0.31908923],
                            [0.66666667, 0.66666667, 0.64723956],
                            [0.0, 0.0, 0.35260054]],
        ids=[0, 1, 1, 1, 1, 2, 2],
        symmorphic=True)

    if 0:
        mp = MonkhorstPackKPoints((5, 5, 1))
        ibz = mp.reduce(sym)
        print(ibz)
        assert (ibz.weight_k > 0.0).all()


@pytest.mark.serial
def test_tolerance():
    a = 1.2
    h = 3.0
    d = 0.001
    tol = 1.1 * d

    # Test length error:
    sym = Symmetries.from_cell([a, a, h, 90, 90, 120])
    assert len(sym) == 24
    sym = Symmetries.from_cell([a + d, a, h, 90, 90, 120], tolerance=tol)
    assert len(sym) == 24

    # Test angle error:
    sym = Symmetries.from_cell([a, a, a])
    assert len(sym) == 48
    sym = Symmetries.from_cell([[a, d, 0], [0, a, 0], [0, 0, a]],
                               tolerance=tol)
    assert len(sym) == 48
