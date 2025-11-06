from math import pi

import numpy as np
from ase import Atoms
from ase.build import add_adsorbate, bulk, fcc111, graphene, molecule, mx2
from ase.lattice.hexagonal import Graphene
from gpaw.benchmark.generate_twisted import make_heterostructure


def system_magic_graphene():
    atoms = graphene(vacuum=5)
    transa_cc = np.array([[29, -30, 0], [59, 29, 0], [0, 0, 1]])
    transb_cc = np.array([[30, -29, 0], [59, 30, 0], [0, 0, 1]])
    atoms = make_heterostructure(atoms, atoms,
                                 transa_cc=transa_cc,
                                 transb_cc=transb_cc,
                                 straina_vv=np.eye(3),
                                 interlayer_dist=3.35)
    return atoms


def system_2188_bl_graphene():
    atoms = graphene(vacuum=5)
    transa_cc = np.array([[27, 13, 0], [14, 27, 0], [0, 0, 1]])
    transb_cc = np.array([[27, 24, 0], [13, 27, 0], [0, 0, 1]])
    atoms = make_heterostructure(atoms, atoms,
                                 transa_cc=transa_cc,
                                 transb_cc=transb_cc,
                                 straina_vv=np.eye(3),
                                 interlayer_dist=3.35)
    return atoms


def system_6000_bl_graphene():
    atoms = graphene(vacuum=5)
    transa_cc = np.array([[23, 45, 0], [-22, 23, 0], [0, 0, 1]])
    transb_cc = np.array([[22, 45, 0], [-23, 22, 0], [0, 0, 1]])
    atoms = make_heterostructure(atoms, atoms,
                                 transa_cc=transa_cc,
                                 transb_cc=transb_cc,
                                 straina_vv=np.eye(3),
                                 interlayer_dist=3.35)
    return atoms


def system_676_bl_graphene():
    atoms = graphene(vacuum=5)
    transa_cc = np.array([[7, -8, 0], [15, 7, 0], [0, 0, 1]])
    transb_cc = np.array([[8, -7, 0], [15, 8, 0], [0, 0, 1]])
    atoms = make_heterostructure(atoms, atoms,
                                 transa_cc=transa_cc,
                                 transb_cc=transb_cc,
                                 straina_vv=np.eye(3),
                                 interlayer_dist=3.35)
    return atoms


def system_H2():
    atoms = molecule('H2')
    atoms.center(vacuum=3)
    return atoms


def system_C60():
    atoms = molecule('C60')
    atoms.center(vacuum=5)
    return atoms


def system_diamond():
    atoms = bulk('C')
    return atoms


def system_MoS2_tube():
    atoms = mx2('MoS2', size=(3, 2, 1))
    atoms.cell[1, 0] = 0
    atoms = atoms.repeat((1, 10, 1))
    p = atoms.positions
    p2 = p.copy()
    L = atoms.cell[1, 1]
    r0 = L / (2 * pi)
    angle = p[:, 1] / L * 2 * pi
    p2[:, 1] = (r0 + p[:, 2]) * np.cos(angle)
    p2[:, 2] = (r0 + p[:, 2]) * np.sin(angle)
    atoms.positions = p2
    atoms.cell = [atoms.cell[0, 0], 0, 0]
    atoms.center(vacuum=6, axis=[1, 2])
    atoms.pbc = True

    return atoms


def system_magbulk():
    atoms = bulk('Fe') * 2
    atoms.set_initial_magnetic_moments([3] * len(atoms))
    return atoms


def system_metalslab():
    slab = fcc111('Al', size=(3, 4, 8), vacuum=6.0)
    return slab


def system_c2db():
    atoms = Atoms(
        symbols='MnVS2',
        pbc=[True, True, False],
        cell=[3.65, 3.97, 1.0],
        scaled_positions=[[0.5, 0.5, 7.22],
                          [0.0, 0.0, 7.49],
                          [0.0, 0.5, 8.70],
                          [0.5, 0.0, 6.00]])
    atoms.center(vacuum=6.0, axis=2)
    atoms.set_initial_magnetic_moments([2, 2, 0, 0])
    return atoms


def opt111b():
    atoms = fcc111('Pt', (2, 2, 6), a=4.00, vacuum=10.0)
    add_adsorbate(atoms, 'O', 2.0, 'fcc')
    return atoms


def lic8():
    ccdist = 1.40
    layerdist = 3.7
    a = ccdist * np.sqrt(3)
    c = layerdist
    Li_gra = Graphene('C', size=(2, 2, 1), latticeconstant={'a': a, 'c': c})
    Li_gra.append('Li')
    Li_gra.positions[-1] = (a / 2, ccdist / 2, layerdist / 2)
    return Li_gra


def vii():
    atoms = mx2('VII', a=4.12, kind='1T', thickness=3.13)
    atoms.center(vacuum=5.0)
    atoms[0].magmom = 3.0
    return atoms


def bi2se3():
    a = 4.138
    c = 28.64
    mu = 0.399
    nu = 0.206
    cell = [[-a / 2, -3**0.5 / 6 * a, c / 3],
            [a / 2, -3**0.5 / 6 * a, c / 3],
            [0.0, 3**0.5 / 3 * a, c / 3]]
    pos = [[mu, mu, mu],
           [-mu, -mu, -mu],
           [0.0, 0.0, 0.0],
           [nu, nu, nu],
           [-nu, -nu, -nu]]
    atoms = Atoms('Bi2Se3', cell=cell, scaled_positions=pos, pbc=True)
    # niggli_reduce(atoms)
    return atoms


def ganfh():
    atoms = Atoms(
        'Ga2N4F4H10',
        cell=[4.859977, 5.863148, 20.105964],
        pbc=True,
        positions=[
            [3.584516, 4.349997, 10.052982],
            [1.154528, 1.418423, 10.052982],
            [4.513082, 4.349997, 11.722455],
            [2.655951, 4.349997, 8.383509],
            [4.824308, 1.418423, 11.681157],
            [2.344725, 1.418423, 8.424807],
            [-0.0, 0.0, 9.251569],
            [-0.0, 2.836845, 9.25156],
            [2.309055, 2.836845, 10.854404],
            [2.309055, 0.0, 10.854404],
            [0.098687, 2.298345, 12.1963],
            [0.098687, 0.5385, 12.1963],
            [2.210368, 0.5385, 7.909665],
            [2.210368, 2.298345, 7.909665],
            [4.018608, 4.349997, 12.605964],
            [3.150424, 4.349997, 7.5],
            [3.842915, 1.418423, 11.37744],
            [3.326117, 1.418423, 8.728524],
            [0.666931, 4.349997, 11.769485],
            [1.642124, 4.349997, 8.33648]])
    return atoms


def pto3li2o3():
    atoms = Atoms(
        'PtO3Li2O3',
        cell=[5.362, 5.362, 32.360676, 90, 90, 120],
        magmoms=[2, 0, 0, 0, 0, 0, 0, 0, 0],
        pbc=[True, True, False],
        positions=[
            [-1.8096749999999993, 3.1344490451872087, 16.180338],
            [1.809675000000001, 3.1344490451872087, 17.360675999999998],
            [1.7426499999999998, 0.0, 17.360675999999998,]
            [-0.9383499999999998, 1.6252698752822565, 17.360675999999998],
            [3.552324999999999, 0.03900647700677462, 16.180338],
            [0.8710568999999999, 1.5867277610969917, 16.180338],
            [-0.06702499999999945, 3.1344490451872087, 15.0],
            [2.6810000000000005, 1.6252698752822565, 15.0],
            [0.0, 0.0, 15.0]])
    return atoms


systems = {'C60': system_C60,
           'diamond': system_diamond,
           'H2': system_H2,
           'MoS2_tube': system_MoS2_tube,
           'C6000': system_6000_bl_graphene,
           'C2188': system_2188_bl_graphene,
           'C676': system_676_bl_graphene,
           'magbulk': system_magbulk,
           'metalslab': system_metalslab,
           'magic_graphene': system_magic_graphene,
           'MnVS2-slab': system_c2db,
           'OPt111b': opt111b,
           'LiC8': lic8,
           'VI2': vii,
           'Bi2Se3': bi2se3,
           'Ga2N4F4H10': ganfh,
           'PtO3Li2O3': pto3li2o3}


def parse_system(name):
    return systems[name]()
