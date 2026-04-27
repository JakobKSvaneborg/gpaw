import numpy as np
import pytest
from pytest import approx

from gpaw.point_groups import PointGroup, SymmetryChecker
from gpaw.point_groups import point_group_names as names


@pytest.mark.serial
@pytest.mark.parametrize('name', names)
def test_for_various_errors(name):
    pg = PointGroup(name)
    print(pg)
    pg.get_normalized_table()


@pytest.mark.serial
@pytest.mark.parametrize('name', names)
def test_translations(name):
    pg = PointGroup(name)
    # Calculate the symmetry representations of p-orbitals:
    x = np.linspace(-5, 5, 11)
    xyz = np.array(np.meshgrid(x, x, x, indexing='ij'))
    r2 = (xyz**2).sum(0)
    checker = SymmetryChecker(pg, center=(5, 5, 5), grid_spacing=1.5)
    for i in range(3):
        func = np.exp(-r2) * xyz[i]
        dct = checker.check_function(func, np.eye(3))
        print(i, dct)
        assert dct['symmetry'] == pg.translations[i]


@pytest.mark.serial
@pytest.mark.parametrize('name', names)
def test_pg(name):
    pg = PointGroup(name)

    # Check that the number of irreducible representation
    # is equal to the number of symmetry
    # transform classes:
    assert len(pg.character_table) == len(pg.character_table[0])

    h = sum(pg.nops)

    # Checks for groups with real character tables:
    if not pg.complex:
        # Check that the sum of squared dimensions of the irreps
        # equals to the number of symmetry elements h
        assert (pg.character_table[:, 0]**2).sum() == h

        # Rows:
        for i, row1 in enumerate(pg.character_table):

            # Check normalization:
            assert (row1**2).dot(pg.nops) == approx(h, 1e-8)

            for j, row2 in enumerate(pg.character_table):

                if i >= j:
                    continue

                # Check orthogonality:
                assert (row1 * row2).dot(pg.nops) == approx(0, abs=1e-8)

        # Columns:
        for i, row1 in enumerate(pg.character_table.T):

            # Check normalization:
            assert (row1**2).sum() == approx(h / pg.nops[i], abs=1e-13)

            for j, row2 in enumerate(pg.character_table.T):

                if i >= j:
                    continue

                # Check orthogonality:
                assert row1.dot(row2) == approx(0, 1e-8)

    # Checks for complex groups:
    else:
        reps = pg.symmetries
        # Rows:
        for i, row1 in enumerate(pg.character_table):

            # Check normalization:
            norm = (row1**2).dot(pg.nops)

            # Real rows:
            if reps[i].find('E') < 0:
                correctnorm = h
            else:  # complex rows
                correctnorm = h * 2

            assert norm == approx(correctnorm, abs=1e-8)

            for j, row2 in enumerate(pg.character_table):

                if i >= j:
                    continue

                # Compare real with real rows and complex with complex rows:
                if ((reps[i].find('E') >= 0 and reps[j].find('E') >= 0)
                    or
                    (reps[i].find('E') < 0 and reps[j].find('E') < 0)):

                    # Check orthogonality:
                    norm = (row1 * row2).dot(pg.nops)
                    assert norm == approx(0, abs=1e-13)
