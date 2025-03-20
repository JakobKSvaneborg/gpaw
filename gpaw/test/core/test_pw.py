from math import pi

import numpy as np
import pytest
from gpaw.core import PWDesc, UGDesc
from gpaw.core.plane_waves import find_reciprocal_vectors
from gpaw.gpu import cupy as cp
from gpaw.gpu.mpi import CuPyMPI
from gpaw.mpi import world
from gpaw.new.c import GPU_AWARE_MPI


@pytest.mark.ci
def test_pw_redist():
    a = 2.5
    pw = PWDesc(ecut=10, cell=[a, a, a], comm=world)
    f1 = pw.empty()
    f1.data[:] = 1.0
    f2 = f1.gather()
    if f2 is not None:
        assert (f2.data == 1.0).all()
        assert f2.desc.comm.size == 1


@pytest.mark.ci
def test_pw_map():
    """Test mapping from 5 to 9 G-vectors."""
    pw5 = PWDesc(ecut=0.51 * (2 * pi)**2,
                 cell=[1, 1, 0.1],
                 dtype=complex)
    pw9 = PWDesc(ecut=1.01 * (2 * pi)**2,
                 cell=[1, 1, 0.1],
                 dtype=complex,
                 comm=world)
    my_G_g, g_r = pw9.map_indices(pw5)
    print(world.rank, my_G_g, g_r)
    expected = {
        1: ([[0, 1, 2, 3, 6]],
            [[0, 1, 2, 3, 4]]),
        2: ([[0, 1, 2, 3], [1]],
            [[0, 1, 2, 3], [4]]),
        4: ([[0, 1, 2], [0], [0], []],
            [[0, 1, 2], [3], [4], []]),
        8: ([[0, 1], [0, 1], [], [0], [], [], [], []],
            [[0, 1], [2, 3], [], [4], [], [], [], []])}
    assert not (my_G_g - expected[world.size][0][world.rank]).any()
    for g, g0 in zip(g_r, expected[world.size][1]):
        assert not (np.array(g) - g0).any()


def grids():
    comm = world
    if not GPU_AWARE_MPI:
        comm = CuPyMPI(comm)
    a = 1.0
    decomp = {1: [[0, 4], [0, 4], [0, 4]],
              2: [[0, 2, 4], [0, 4], [0, 4]],
              4: [[0, 2, 4], [0, 2, 4], [0, 4]],
              8: [[0, 1, 2, 3, 4], [0, 2, 4], [0, 4]]}[world.size]
    grid = UGDesc(cell=[a, a, a], size=(4, 4, 4), comm=comm, decomp=decomp)
    gridc = grid.new(dtype=complex)

    g1 = grid.empty()
    g1.data[:] = 1.0

    g2 = grid.empty()
    g2.data[:] = 1.0
    g2.data += [0, 1, 0, -1]

    g3 = gridc.empty()
    g3.data[:] = 1.0

    g4 = gridc.empty()
    g4.data[:] = 1.0
    g4.data += [0, 1, 0, -1]

    g5 = gridc.empty()
    g5.data[:] = 1.0 + 1.0j
    g5.data += [0, 1, 0, -1]

    return [g1, g2, g3, g4, g5]


@pytest.mark.gpu
@pytest.mark.parametrize('xp', [np, cp])
@pytest.mark.parametrize('grid', grids())
def test_pw_integrate(xp, grid):
    g = grid
    a = g.desc.cell[0, 0]
    ecut = 0.5 * (2 * np.pi / a)**2 * 1.01
    if xp is cp:
        g = g.to_xp(cp)
    pw = PWDesc(cell=g.desc.cell, dtype=g.desc.dtype,
                ecut=ecut, comm=g.desc.comm)
    f = g.fft(pw=pw)

    gg = g.new()
    gg.scatter_from(f.gather(broadcast=True)
                    .ifft(grid=g.desc.new(comm=None)))
    assert xp.allclose(g.data, gg.data, rtol=1e-14)

    i1 = g.integrate()
    i2 = f.integrate()
    assert i1 == i2
    assert np.dtype(type(i1)) == g.desc.dtype

    i1 = g.integrate(g)
    i2 = f.integrate(f)
    assert i1 == i2
    assert np.dtype(type(i1)) == g.desc.dtype

    g1 = g.desc.empty(1, xp=xp)
    g1.data[:] = g.data
    m1 = g1.matrix_elements(g1)
    assert i1 == m1.data.item()

    f1 = f.desc.empty(1, xp=xp)
    f1.data[:] = f.data
    m2 = f1.matrix_elements(f1)
    assert i2 == pytest.approx(m2.data[0, 0].item(), abs=1e-13)


def test_grr():
    from ase.units import Bohr, Ha
    grid = UGDesc(cell=[2 / Bohr, 2 / Bohr, 2.737166 / Bohr],
                  size=(9, 9, 12),
                  comm=world)
    pw = PWDesc(ecut=340 / Ha, cell=grid.cell, comm=world)
    print(pw.G_plus_k_Gv.shape)
    from gpaw.grid_descriptor import GridDescriptor
    from gpaw.pw.descriptor import PWDescriptor
    g = GridDescriptor((9, 9, 12), [2 / Bohr, 2 / Bohr, 2.737166 / Bohr])
    p = PWDescriptor(340 / Ha, g)
    print(p.get_reciprocal_vectors().shape)
    assert (p.get_reciprocal_vectors() == pw.G_plus_k_Gv).all()


def test_find_g():
    G, e, i = find_reciprocal_vectors(0.501 * (2 * pi)**2,
                                      np.eye(3))
    assert i.T.tolist() == [[0, 0, 0],
                            [0, 0, 1],
                            [0, 0, -1],
                            [0, 1, 0],
                            [0, -1, 0],
                            [1, 0, 0],
                            [-1, 0, 0]]
    G, e, i = find_reciprocal_vectors(0.501 * (2 * pi)**2,
                                      np.eye(3),
                                      dtype=float)
    assert i.T.tolist() == [[0, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0],
                            [1, 0, 0]]
    G, e, i = find_reciprocal_vectors(0.5 * (2 * pi)**2,
                                      np.eye(3),
                                      kpt=np.array([0.1, 0, 0]))
    assert i.T.tolist() == [[0, 0, 0],
                            [-1, 0, 0]]


def test_random():
    pw = PWDesc(ecut=20, cell=[1, 1, 1], comm=world)
    a = pw.empty(2)
    a.randomize()
    assert world.rank != 0 or (a.data[:, 0].imag == 0.0).all()
