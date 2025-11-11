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
    atoms.pbc = [True, False, False]

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
    atoms.cell[:] = [[1, 0, -1], [0, 1, -1], [1, 1, 1]] @ atoms.cell
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
    # From /home/niflheim2/cmr/C2DB-ASR/ICSD-COD/not_converged/Li2PtO6/
    atoms = Atoms(
        'PtO3Li2O3',
        cell=[5.362, 5.362, 32.360676, 90, 90, 120],
        magmoms=[2, 0, 0, 0, 0, 0, 0, 0, 0],
        pbc=[True, True, False],
        positions=[
            [-1.8096749999999993, 3.1344490451872087, 16.180338],
            [1.809675000000001, 3.1344490451872087, 17.360675999999998],
            [1.7426499999999998, 0.0, 17.360675999999998],
            [-0.9383499999999998, 1.6252698752822565, 17.360675999999998],
            [3.552324999999999, 0.03900647700677462, 16.180338],
            [0.8710568999999999, 1.5867277610969917, 16.180338],
            [-0.06702499999999945, 3.1344490451872087, 15.0],
            [2.6810000000000005, 1.6252698752822565, 15.0],
            [0.0, 0.0, 15.0]])
    return atoms


def erge():
    # From /home/niflheim2/cmr/C2DB-ASR/ICSD-COD/lanthanides/ErGe/
    atoms = Atoms(
        'ErGe',
        cell=[3.911, 4.091, 1.0],
        magmoms=[0, 2],
        pbc=[True, True, False],
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 1.124]])
    atoms.center(vacuum=5.5, axis=2)
    return atoms


def as4crsi2():
    atoms = Atoms(
        'As4CrSi2',
        cell=[3.81, 3.81, 1.0, 90, 90, 120],
        magmoms=[0, 0, 0, 0, 4, 0, 0],
        pbc=[True, True, False],
        scaled_positions=[
            [0, 0, -4.66],
            [0, 0, 4.66],
            [1 / 3, 2 / 3, -1.12],
            [2 / 3, 1 / 3, 1.12],
            [0, 0, 0],
            [2 / 3, 1 / 3, 3.54],
            [1 / 3, 2 / 3, -3.54]])
    atoms.center(vacuum=6.5, axis=2)
    return atoms


def v3cl6():
    a = 6.339
    d = 1.331
    atoms = Atoms(
        'V3Cl6',
        cell=[a, a, 1, 90, 90, 60],
        pbc=[1, 1, 0],
        scaled_positions=[
            [0, 0, 0],
            [1 / 3, 1 / 3, 0],
            [2 / 3, 2 / 3, 0],
            [0, 2 / 3, d],
            [0, 1 / 3, -d],
            [1 / 3, 0, d],
            [1 / 3, 2 / 3, -d],
            [2 / 3, 1 / 3, d],
            [2 / 3, 0, -d]])
    atoms.center(axis=2, vacuum=5)
    m = 3.0
    magmoms = np.zeros((9, 3))
    magmoms[0] = [m, 0, 0]
    magmoms[1] = [-m / 2, m * 3**0.5 / 2, 0]
    magmoms[2] = [-m / 2, -m * 3**0.5 / 2, 0]
    atoms._magmoms = magmoms
    return atoms


def mn2o2():
    a = 4.5155
    b = a / 2
    atoms = Atoms(
        'Mn2O2',
        cell=[[a, b, b], [b, a, b], [b, b, a]],
        pbc=True,
        positions=[[0, 0, 0],
                   [a, a, a],
                   [b, b, b],
                   [a + b, a + b, a + b]],
        magmoms=[1, -1, 0, 0])
    atoms.cell[:] = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]] @ atoms.cell
    return atoms


def ti2br6():
    atoms = Atoms(
        'Ti2Br6',
        [[8.1142, 6.4498, 6.8322],
         [4.8964, 1.0125, 0.0002],
         [3.9117, 2.8044, 1.5473],
         [9.0989, 4.6578, 5.2852],
         [6.9947, 4.7843, 1.4810],
         [6.0159, 2.6779, 5.3514],
         [6.0215, 6.6423, 5.3382],
         [6.9891, 0.8199, 1.4942]],
        cell=[[6.518707, 0.000000, 0.000000],
              [3.264878, 5.606928, 0.000000],
              [3.227015, 1.855287, 6.832419]],
        pbc=True)
    return atoms


def fe8o8():
    atoms = Atoms(
        'Fe8O8',
        cell=[[-0.017249, 4.052906, 4.049397],
              [4.510540, -0.477776, 4.510540],
              [4.049397, 4.052906, -0.017249]],
        positions=[
            [0.0000, -0.0000, 0.0000],
            [-2.0333, -0.0000, 2.0334],
            [-0.0086, 2.0264, 2.0247],
            [-2.0420, 2.0264, 4.0581],
            [-2.2639, 2.2653, -0.2306],
            [-4.2972, 2.2653, 1.8028],
            [-2.2726, 4.2918, 1.7941],
            [-4.3058, 4.2918, 3.8275],
            [2.1356, 1.9070, 2.1356],
            [0.1024, 1.9070, 4.1690],
            [2.1271, 3.9335, 4.1603],
            [0.0937, 3.9335, 6.1937],
            [-0.1282, 4.1723, 1.9051],
            [-2.1615, 4.1723, 3.9384],
            [-0.1368, 6.1988, 3.9298],
            [-2.1702, 6.1988, 5.9631]],
        magmoms=[2] * 8 + [0] * 8,
        pbc=True)
    return atoms


def c72():
    return graphene(size=(6, 6, 1), vacuum=5.0)


systems = {'H2': system_H2,
           'C60': system_C60,
           'MoS2_tube': system_MoS2_tube,
           'C6000': system_6000_bl_graphene,
           'C2188': system_2188_bl_graphene,
           'C676': system_676_bl_graphene,
           'metalslab': system_metalslab,
           'magic_graphene': system_magic_graphene,
           'MnVS2-slab': system_c2db,
           'OPt111b': opt111b,
           'diamond': system_diamond,
           'magbulk': system_magbulk,
           'LiC8': lic8,
           'VI2': vii,
           'Bi2Se3': bi2se3,
           'Ga2N4F4H10': ganfh,
           'PtO3Li2O3': pto3li2o3,
           'ErGe': erge,
           'As4CrSi2': as4crsi2,
           'V3Cl6': v3cl6,
           'Mn2O2': mn2o2,
           'Ti2Br6': ti2br6,
           'Fe8O8': fe8o8,
           'C72': c72}


def parse_system(name):
    return systems[name]()
