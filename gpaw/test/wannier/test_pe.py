"""Poly-ethylene 1-d chain."""
import pytest
import numpy as np
from ase.neighborlist import neighbor_list

from gpaw import GPAW
from gpaw.wannier import calculate_overlaps


def check(atoms, rcut=0.8):
    """Make sure all Wannier function centers have two neighbors."""
    atoms.wrap()
    i = neighbor_list('i', atoms, rcut)
    nneighbors = np.bincount(i)
    for symbol, nn in zip(atoms.symbols, nneighbors):
        if symbol == 'X':
            assert nn == 2


@pytest.mark.wannier
@pytest.mark.serial
def test_pe_w90(gpw_files, in_tmp_dir, wannier90):
    calc = GPAW(gpw_files['c2h4_pw_nosym'])
    o = calculate_overlaps(calc, n2=6, nwannier=6,
                           projections={'C': 's', 'H': 's'})
    w = o.localize_w90('pe')
    check(w.centers_as_atoms())


@pytest.mark.wannier
@pytest.mark.serial
def test_pe_er(gpw_files):
    calc = GPAW(gpw_files['c6h12_pw'])
    o = calculate_overlaps(calc, n2=3 * 6, nwannier=18)
    w = o.localize_er()
    check(w.centers_as_atoms())
