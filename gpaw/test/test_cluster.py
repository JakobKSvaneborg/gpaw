from math import sqrt

from ase import Atoms
from ase.build import fcc111

from gpaw.cluster import Cluster
from gpaw.mpi import world
import pytest


def test_cluster():
    R = 2.0
    CO = Atoms('CO', [(1, 0, 0), (1, 0, R)])

    CO.rotate(90, 'y')
    assert CO.positions[1, 0] == pytest.approx(R, abs=1e-10)

    # translate
    CO.translate(-CO.get_center_of_mass())
    p = CO.positions.copy()
    for i in range(2):
        assert p[i, 1] == pytest.approx(0, abs=1e-10)
        assert p[i, 2] == pytest.approx(0, abs=1e-10)

    # rotate the nuclear axis to the direction (1,1,1)
    CO.rotate(p[1] - p[0], (1, 1, 1))
    q = CO.positions.copy()
    for c in range(3):
        assert q[0, c] == pytest.approx(p[0, 0] / sqrt(3), abs=1e-10)
        assert q[1, c] == pytest.approx(p[1, 0] / sqrt(3), abs=1e-10)

    # minimal box
    b = 4.0
    CO = Cluster(['C', 'O'], [(1, 0, 0), (1, 0, R)])
    CO.minimal_box(b)
    cc = CO.get_cell()
    for c in range(3):
        width = 2 * b
        if c == 2:
            width += R
        assert cc[c, c] == pytest.approx(width, abs=1e-10)

    # minimal box, ensure multiple of 4
    h = .13
    b = [2, 3, 4]
    CO.minimal_box(b, h=h)
    cc = CO.get_cell()
    for c in range(3):
        # print "cc[c,c], cc[c,c] / h % 4 =", cc[c, c], cc[c, c] / h % 4
        for a in CO:
            print(a.symbol, b[c], a.position[c], cc[c, c] - a.position[c])
            assert a.position[c] > b[c]
        assert cc[c, c] / h % 4 == pytest.approx(0.0, abs=1e-10)

    # I/O
    fxyz = 'CO.xyz'
    fpdb = 'CO.pdb'

    cell = [2., 3., R + 2.]
    CO.set_cell(cell, scale_atoms=True)
    world.barrier()
    CO.write(fxyz)
    world.barrier()
    CO_b = Cluster(filename=fxyz)
    assert len(CO) == len(CO_b)
    offdiagonal = CO_b.get_cell().sum() - CO_b.get_cell().diagonal().sum()
    assert offdiagonal == 0.0

    world.barrier()
    CO.write(fpdb)

    # read xyz files with additional info
    read_with_additional = True
    if read_with_additional:
        if world.rank == 0:
            f = open(fxyz, 'w')
            print("""2

    C 0 0 0. 1 2 3
    O 0 0 1. 6. 7. 8.""", file=f)
            f.close()

        world.barrier()

        CO = Cluster(filename=fxyz)


def test_minimal_box_mixed_pbc():

    atoms = Cluster(Atoms('H'))
    atoms.center(vacuum=2)
    atoms.pbc = [0, 1, 1]
    cell0 = atoms.cell.copy()

    box = 3
    atoms.minimal_box(box)
    assert atoms.cell[0, 0] == 2 * box
    assert atoms.cell[1:, 1:] == pytest.approx(cell0[1:, 1:])

    atoms.cell[1, 1] = 3
    atoms.minimal_box(box, h='periodic')

    assert atoms.cell[0, 0] == pytest.approx(7)

    # testing non orthogonal uint sell
    a = 3.92
    vac = 2
    atoms = Cluster(fcc111('Pt', (1, 1, 1), a=a, vacuum=vac))
    atoms.pbc = [1, 1, 0]
    cell0 = atoms.cell
    atoms.minimal_box(box)

    assert atoms.cell[2, 2] == 2 * box
    assert atoms.cell[:1, :1] == pytest.approx(cell0[:1, :1])

    atoms.cell[0, 0] = 4
    atoms.cell[1, 1] = 3
    atoms.minimal_box(box, h='periodic')

    assert atoms.cell[2, 2] == pytest.approx(7)
