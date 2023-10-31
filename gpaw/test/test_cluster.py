from math import sqrt

from ase import Atoms

from gpaw.cluster import Cluster
from gpaw.mpi import world
from gpaw.test import equal


def test_cluster(in_tmp_dir):
    R = 2.0
    CO = Atoms('CO', [(1, 0, 0), (1, 0, R)])

    CO.rotate(90, 'y')
    equal(CO.positions[1, 0], R, 1e-10)

    # translate
    CO.translate(-CO.get_center_of_mass())
    p = CO.positions.copy()
    for i in range(2):
        equal(p[i, 1], 0, 1e-10)
        equal(p[i, 2], 0, 1e-10)

    # rotate the nuclear axis to the direction (1,1,1)
    CO.rotate(p[1] - p[0], (1, 1, 1))
    q = CO.positions.copy()
    for c in range(3):
        equal(q[0, c], p[0, 0] / sqrt(3), 1e-10)
        equal(q[1, c], p[1, 0] / sqrt(3), 1e-10)

    # minimal box
    b = 4.0
    CO = Cluster(['C', 'O'], [(1, 0, 0), (1, 0, R)])
    CO.minimal_box(b)
    cc = CO.get_cell()
    for c in range(3):
        width = 2 * b
        if c == 2:
            width += R
        equal(cc[c, c], width, 1e-10)

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
        equal(cc[c, c] / h % 4, 0.0, 1e-10)

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
