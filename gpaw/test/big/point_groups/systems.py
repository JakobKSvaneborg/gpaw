import math

import numpy as np
from ase import Atom, Atoms
from ase.build import molecule

systems = {}

name = 'dichloroallene'
system = Atoms()
CC = 1.33  # C-C bond length
CH = 1.1  # C-H bond length
CCl = 1.74  # C-Cl bond length
system.append(Atom('C', position=[0, 0, 0]))
system.append(Atom('C', position=[CC, 0, 0]))
system.append(Atom('C', position=[-CC, 0, 0]))
c = 1 / math.sqrt(3.0)
system.append(Atom('Cl', position=[CC + c * CCl, c * CCl, c * CCl]))
system.append(Atom('H', position=[CC + c * CH, -c * CH, -c * CH]))
system.append(Atom('Cl', position=[-CC - c * CCl, -c * CCl, c * CCl]))
system.append(Atom('H', position=[-CC - c * CH, +c * CH, -c * CH]))
systems['C2'] = (name, system)

name = 'H2O'
a = 8.0
h = 0.2
system = molecule(name)
systems['C2v'] = (name, system)

name = 'NH3'
deg2rad = 2 * math.pi / 360.
NH = 1.008  # N-H bond length
alpha = 107.5 * deg2rad  # H-N-H angle
system = Atoms()
system.append(Atom('N', position=[0, 0, 0]))
x = NH * math.sin(alpha / 2.) / math.cos(math.pi / 6.)
y = 0.
z = math.sqrt(NH**2 - x**2)
atom0 = Atoms()
atom0.append(Atom('H', position=[x, y, z]))
beta = 2 * math.pi / 3.
rotmat = np.array([[math.cos(beta), -math.sin(beta), 0.],
                   [+math.sin(beta), math.cos(beta), 0.], [0., 0., 1.]])
for i in range(3):
    atom0[0].position = rotmat.dot(atom0[0].position)
    system.append(atom0[0])
systems['C3v'] = (name, system)

name = 'allene'
system = Atoms()
CC = 1.33  # C-C bond length
CH = 1.1  # C-H bond length
system.append(Atom('C', position=[0, 0, 0]))
system.append(Atom('C', position=[0, 0, CC]))
system.append(Atom('C', position=[0, 0, -CC]))
c = 1 / np.sqrt(2.0)
system.append(Atom('H', position=[0., c * CH, CC + c * CH]))
system.append(Atom('H', position=[0., -c * CH, CC + c * CH]))
system.append(Atom('H', position=[c * CH, 0., -CC - c * CH]))
system.append(Atom('H', position=[-c * CH, 0., -CC - c * CH]))
systems['D2d'] = (name, system)

name = 'BF3'
deg2rad = 2 * math.pi / 360.
BF = 1.313  # B-F bond length
alpha = 120. * deg2rad  # F-B-F angle
system = Atoms()
system.append(Atom('B', position=[0, 0, 0]))
x = BF * math.sin(alpha / 2.) / math.cos(math.pi / 6.)
y = 0.
z = math.sqrt(BF**2 - x**2)
atom0 = Atoms()
atom0.append(Atom('F', position=[x, y, z]))
beta = 2 * math.pi / 3.
rotmat = np.array([[math.cos(beta), -math.sin(beta), 0.],
                   [+math.sin(beta), math.cos(beta), 0.], [0., 0., 1.]])
for i in range(3):
    atom0[0].position = rotmat.dot(atom0[0].position)
    system.append(atom0[0])
systems['D3h'] = (name, system)

name = 'ferrocene'
system = Atoms()
system.append(Atom('Fe', position=[0, 0, 0]))
height = 1.68  # distance of cyclopentadienyl from Fe
CC = 1.425  # C-C distance
CH = 1.080  # C-H distance
FeC_xy = CC / (2. * math.sin(2 * math.pi / 5. / 2.)
               )  # distance of C from z axis
chirality_angle = 0. * (2 * math.pi / 360.)
# Top ring:
for i in range(5):
    x = FeC_xy * math.cos(i * 2 * math.pi / 5. + chirality_angle / 2.)
    y = FeC_xy * math.sin(i * 2 * math.pi / 5. + chirality_angle / 2.)
    z = height
    system.append(Atom('C', position=[x, y, z]))

    x = (FeC_xy + CH) * math.cos(i * 2 * math.pi / 5. + chirality_angle / 2.)
    y = (FeC_xy + CH) * math.sin(i * 2 * math.pi / 5. + chirality_angle / 2.)
    z = height
    system.append(Atom('H', position=[x, y, z]))
# Bottom ring:
for i in range(5):
    x = FeC_xy * math.cos(i * 2 * math.pi / 5. - chirality_angle / 2.)
    y = FeC_xy * math.sin(i * 2 * math.pi / 5. - chirality_angle / 2.)
    z = -height
    system.append(Atom('C', position=[x, y, z]))
    x = (FeC_xy + CH) * math.cos(i * 2 * math.pi / 5. - chirality_angle / 2.)
    y = (FeC_xy + CH) * math.sin(i * 2 * math.pi / 5. - chirality_angle / 2.)
    z = -height
    system.append(Atom('H', position=[x, y, z]))
systems['D5h'] = (name, system)

name = 'ferrocene-chiral'
system = Atoms()
system.append(Atom('Fe', position=[0, 0, 0]))
height = 1.68  # distance of cyclopentadienyl from Fe
CC = 1.425  # C-C distance
CH = 1.080  # C-H distance
FeC_xy = CC / (2. * math.sin(2 * math.pi / 5. / 2.)
               )  # distance of C from z axis
chirality_angle = 14. * (2 * math.pi / 360.)
# Top ring:
for i in range(5):
    x = FeC_xy * math.cos(i * 2 * math.pi / 5. + chirality_angle / 2.)
    y = FeC_xy * math.sin(i * 2 * math.pi / 5. + chirality_angle / 2.)
    z = height
    system.append(Atom('C', position=[x, y, z]))

    x = (FeC_xy + CH) * math.cos(i * 2 * math.pi / 5. + chirality_angle / 2.)
    y = (FeC_xy + CH) * math.sin(i * 2 * math.pi / 5. + chirality_angle / 2.)
    z = height
    system.append(Atom('H', position=[x, y, z]))
# Bottom ring:
for i in range(5):
    x = FeC_xy * math.cos(i * 2 * math.pi / 5. - chirality_angle / 2.)
    y = FeC_xy * math.sin(i * 2 * math.pi / 5. - chirality_angle / 2.)
    z = -height
    system.append(Atom('C', position=[x, y, z]))

    x = (FeC_xy + CH) * math.cos(i * 2 * math.pi / 5. - chirality_angle / 2.)
    y = (FeC_xy + CH) * math.sin(i * 2 * math.pi / 5. - chirality_angle / 2.)
    z = -height
    system.append(Atom('H', position=[x, y, z]))
systems['D5'] = (name, system)

name = 'dodecaborate'
# Icosahedron params:
BB = 1.8  # B-B bond length
BH = 1.2  # B-H bond length
phi = 0.5 * (1. + math.sqrt(5.))
l = 1.0  # streching parameter
P = BB / 2.  # edge length parameter
ico = np.array([[0., l, phi], [0., l, -phi], [0., -l, phi], [0., -l, -phi],
                [l, phi, 0.], [l, -phi, 0.], [-l, phi, 0.], [-l, -phi, 0.],
                [phi, 0., l], [-phi, 0., l], [phi, 0., -l], [-phi, 0., -l]
                ]) * P
system = Atoms()
for corner in ico:
    system.append(Atom('B', position=corner))
    Hpos = corner + BH * corner / np.linalg.norm(corner)
    system.append(Atom('H', position=Hpos))
R = system.positions
system.rotate(R[5] - R[3], 'z')
system.rotate(36, 'z')
systems['Ih'] = (name, system)

name = 'dodecaborate'
# Icosahedron params:
BB = 1.8  # B-B bond length
BH = 1.2  # B-H bond length
phi = 0.5 * (1. + math.sqrt(5.))
l = 1.0  # streching parameter
P = BB / 2.  # edge length parameter
ico = np.array([[0., l, phi], [0., l, -phi], [0., -l, phi], [0., -l, -phi],
                [l, phi, 0.], [l, -phi, 0.], [-l, phi, 0.], [-l, -phi, 0.],
                [phi, 0., l], [-phi, 0., l], [phi, 0., -l], [-phi, 0., -l]
                ]) * P
system = Atoms()
for corner in ico:
    system.append(Atom('B', position=corner))
    Hpos = corner + BH * corner / np.linalg.norm(corner)
    system.append(Atom('H', position=Hpos))
systems['Ico'] = (name, system)

name = 'F6S'
system = Atoms()
SF = 1.564  # S-F bond length
system.append(Atom('S', position=[0, 0, 0]))
system.append(Atom('F', position=[0., 0., SF]))
system.append(Atom('F', position=[0., 0., -SF]))
system.append(Atom('F', position=[0., SF, 0.]))
system.append(Atom('F', position=[0., -SF, 0.]))
system.append(Atom('F', position=[SF, 0., 0.]))
system.append(Atom('F', position=[-SF, 0., 0.]))
systems['Oh'] = (name, system)

name = 'P4'
PP = 2.234  # P-P bond length
alpha = 60. * deg2rad
system = Atoms()
system.append(Atom('P', position=[0, 0, 0]))
x = PP * math.sin(alpha / 2.) / math.cos(math.pi / 6.)
y = 0.
z = -math.sqrt(PP**2 - x**2)
atom0 = Atoms()
atom0.append(Atom('P', position=[x, y, z]))
beta = 2 * math.pi / 3.
rotmat = np.array([[math.cos(beta), -math.sin(beta), 0.],
                   [+math.sin(beta), math.cos(beta), 0.], [0., 0., 1.]])
for i in range(3):
    atom0[0].position = rotmat.dot(atom0[0].position)
    system.append(atom0[0])
system.rotate(180, 'z')
systems['Td'] = (name, system)

name = 'Mg-H2O-6'
MgO = 2.09  # Mg-O bond length
atoms = Atoms()
atoms.append(Atom('Mg', position=[0, 0, 0]))
h2o = molecule('H2O')
opos = h2o[0].position.copy()
for atom in h2o:
    atom.position -= opos
for atom in h2o:
    atom.position += np.array([0., 0., -MgO])
for atom in h2o:
    atoms.append(atom)
    atom.position *= -1
    atoms.append(atom)
for atom in h2o:
    atom.position = np.array(
        [atom.position[1], atom.position[0], atom.position[2]])
    atom.position = np.array(
        [atom.position[2], atom.position[1], atom.position[0]])
    atoms.append(atom)
    atom.position *= -1
    atoms.append(atom)
for atom in h2o:
    atom.position = np.array(
        [atom.position[0], atom.position[2], atom.position[1]])
    atom.position = np.array(
        [atom.position[1], atom.position[0], atom.position[2]])
    atoms.append(atom)
    atom.position *= -1
    atoms.append(atom)
system = atoms
system.rotate([1, 1, 1], 'z')
system.rotate(75, 'z')
systems['Th'] = (name, system)

for group_name, (name, system) in systems.items():
    system.write(f'{group_name}-{name}.xyz')
