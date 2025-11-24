import pytest

from gpaw.core import PWDesc, UGDesc
from gpaw.core.matrix import Matrix


def func(f):
    """Operator for matrix elements."""
    g = f.copy()
    g.data *= 2.3
    return g


# TODO: test also UGArray
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('nbands', [1, 7, 21])
@pytest.mark.parametrize('function', [None, func])
def test_me(domain_band_comms, dtype, nbands, function):
    domain_comm, band_comm = domain_band_comms
    a = 2.5
    n = 20
    grid = UGDesc(cell=[a, a, a], size=(n, n, n))
    desc = PWDesc(ecut=50, cell=grid.cell)
    desc = desc.new(comm=domain_comm, dtype=dtype)
    f = desc.empty(nbands, comm=band_comm)
    f.randomize()

    M = f.matrix_elements(f, function=function)
    out = Matrix(nbands, nbands, dist=(band_comm, -1, 1), dtype=dtype)
    out.data[:] = 1e308  # will overflow when multiplied by 2
    f.matrix_elements(f, function=function, out=out)

    f1 = f.gathergather()
    M2 = M.gather()
    if f1 is not None:
        M1 = f1.matrix_elements(f1, function=function)
        M1.tril2full()
        M2.tril2full()
        dM = M1.data - M2.data
        assert abs(dM).max() < 1e-11

    if function is None:
        g = f.new()
        g.randomize()
        M = f.matrix_elements(g)

        f1 = f.gathergather()
        g1 = g.gathergather()
        M2 = M.gather()
        if f1 is not None:
            M1 = f1.matrix_elements(g1)
            M1.tril2full()
            M2.tril2full()
            dM = M1.data - M2.data
            assert abs(dM).max() < 1e-11
