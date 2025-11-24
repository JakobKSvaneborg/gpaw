import numpy as np
import pytest

from gpaw.core.atom_arrays import AtomArraysLayout, AtomDistribution
from gpaw.gpu import cupy as cp


def test_aa_to_full():
    d = np.array([[1, 2, 4],
                  [2, 3, 5],
                  [4, 5, 6]], dtype=float)
    a = AtomArraysLayout([(3, 3)]).empty()
    a[0][:] = d
    p = a.to_lower_triangle()
    assert (p[0] == [1, 2, 3, 4, 5, 6]).all()
    assert (p.to_full()[0] == d).all()


def test_scatter_from(comm):
    N = 9
    atomdist1 = AtomDistribution([0] * N, comm)
    b1 = AtomArraysLayout([(3, 3)] * N, atomdist=atomdist1).empty(2)
    for a, b_sii in b1.items():
        assert comm.rank == 0
        b_sii[0] = a
        b_sii[1] = 2 * a
    b2 = b1.gather()
    if comm.rank == 0:
        assert (b1.data == b2.data).all()
    atomdist3 = AtomDistribution.from_number_of_atoms(N, comm)
    b3 = b1.layout.new(atomdist=atomdist3).empty(2)
    b3.scatter_from(b2.data if b2 is not None else None)
    for a, b_sii in b3.items():
        assert (b_sii[0] == a).all()
        assert (b_sii[1] == 2 * a).all()


def test_gather(comm):
    """Two atoms on rank-1."""
    r = min(1, comm.size - 1)
    ranks = [r, r]
    atomdist = AtomDistribution(ranks, comm)
    D_asii = AtomArraysLayout([(1, 1)] * 2, atomdist=atomdist).empty(1)
    if comm.rank == r:
        D_asii[0][:] = 1
        D_asii[1][:] = 2
    D2_asii = D_asii.gather(broadcast=True)
    assert D2_asii.data.shape == (1, 2)
    for a, D_sii in D2_asii.items():
        assert D_sii[0, 0, 0] == a + 1


@pytest.mark.gpu
@pytest.mark.parametrize('xp', [np, cp])
def test_P_ani_dH_aii(domain_band_comms, xp):
    domain_comm, band_comm = domain_band_comms
    ni_a = [2, 3, 4, 17]
    dH_asii = AtomArraysLayout([(n, n) for n in ni_a],
                               atomdist=domain_comm,
                               xp=xp).empty(1)
    dH_asii.data[:] = 1.0
    P_ani = AtomArraysLayout(ni_a,
                             dtype=complex,
                             atomdist=domain_comm,
                             xp=xp).empty(
        10, comm=band_comm)
    P_ani.data[:] = 1.0j
    out_ani = P_ani.new()
    P_ani.block_diag_multiply(dH_asii, out_ani, index=0)
    for a, out_ni in out_ani.items():
        assert (out_ni == ni_a[a] * 1.0j).all()
