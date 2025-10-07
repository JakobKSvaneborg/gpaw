"""Test exact exchange for 20 small molecules.

Compare results to::

  S. Kurth, J. P. Perdew, and P. Blaha
  Molecular and Soild-State Tests of Density Functional
  Approximations: LSD, GGAs, and Meta-GGAs
  International Journal of Quantum Chemistry, Vol. 75, 889-909, 1999

"""

import ase.db
from ase import Atoms
from ase.build import molecule

from gpaw import GPAW, PW
from gpaw.hybrids.energy import non_self_consistent_energy

# flake8: noqa
# Experimental, and calculated bindinglengths of 16 diatomic molecules
# of the G2-1 test set. In Angstroms.
# Data from reference [3].
diatomic = {
    #System Expt. PBEVASP PBEG03 PBE0VASP PBE0G03
    'BeH': (1.343, 1.354, 1.353, 1.350, 1.348),
    'CH' : (1.120, 1.136, 1.136, 1.124, 1.124),
    'Cl2': (1.988, 1.999, 2.004, 1.973, 1.978),
    'ClF': (1.628, 1.648, 1.650, 1.614, 1.617),
    'ClO': (1.570, 1.576, 1.577, 1.554, 1.555),
    'CN' : (1.172, 1.173, 1.174, 1.159, 1.159),
    'CO' : (1.128, 1.136, 1.135, 1.122, 1.122),
    'F2' : (1.412, 1.414, 1.413, 1.377, 1.376),
    'FH' : (0.917, 0.932, 0.930, 0.919, 0.918),
    'HCl': (1.275, 1.287, 1.288, 1.276, 1.278),
    'Li2': (2.673, 2.728, 2.728, 2.727, 2.727),
    'LiF': (1.564, 1.583, 1.575, 1.571, 1.561),
    'LiH': (1.595, 1.604, 1.604, 1.602, 1.597),
    'N2' : (1.098, 1.103, 1.102, 1.089, 1.089),
    'O2' : (1.208, 1.218, 1.218, 1.193, 1.192),
    'Na2': (3.079, 3.087, 3.076, 3.086, 3.086)}

# exchange-only atomization energies in kcal / mol (= 43.364 meV)
# All values evaluated with PBE xc-orbitals and densities at
# experimental geometries. (from [1]).
ex_atomization = {
    # Molec   exact   LSD    PBE    RPBE   BLYP
    'H2'  : (  84.0,  81.5,  84.8,  85.8,  85.4),
    'LiH' : (  33.9,  33.6,  36.9,  36.8,  36.2),
    'CH4' : ( 327.2, 369.9, 336.0, 326.9, 331.2),
    'NH3' : ( 199.5, 255.0, 227.4, 218.9, 222.6),
    'OH'  : (  67.3,  96.2,  84.5,  80.9,  82.7),
    'H2O' : ( 154.6, 212.9, 183.9, 176.4, 180.5),
    'HF'  : (  96.1, 136.1, 117.1, 112.6, 115.4),
    'Li2' : (   3.5,   6.5,   6.4,   6.7,   3.9),
    'LiF' : (  86.8, 129.7, 116.5, 110.8, 113.6),
    'Be2' : ( -11.0,   9.4,   3.1,   1.2,   1.6),
    'C2H2': ( 290.6, 382.7, 333.0, 318.5, 325.5),
    'C2H4': ( 423.9, 517.7, 456.5, 439.6, 447.4),
    'HCN' : ( 194.5, 294.0, 256.1, 243.5, 249.1),
    'CO'  : ( 169.2, 261.9, 224.0, 213.1, 218.7),
    'N2'  : ( 110.2, 211.4, 184.1, 173.6, 177.6),
    'NO'  : (  45.6, 156.9, 122.8, 112.5, 117.0),
    'O2'  : (  24.9, 147.5, 104.4,  94.1,  99.3),
    'F2'  : ( -43.3,  64.0,  32.5,  24.7,  28.8),
    'P2'  : (  31.8,  98.4,  73.1,  66.1,  70.1),
    'Cl2' : (  15.5,  68.2,  39.8,  33.7,  37.0)}

# Experimental bondlengths:
bondlengths = {'H2': 0.741,
               'OH': 0.970,
               'HF': 0.9168,
               'NO': 1.154,
               'P2': 1.893}
bondlengths.update((name, d[0]) for name, d in diatomic.items())

extra = {
    'CH4': ('CH4', [(0.0000, 0.0000, 0.0000),
                    (0.6276, 0.6276, 0.6276),
                    (0.6276, -0.6276, -0.6276),
                    (-0.6276, 0.6276, -0.6276),
                    (-0.6276, -0.6276, 0.6276)]),
    'NH3': ('NH3', [(0.0000, 0.0000, 0.0000),
                    (0.0000, -0.9377, -0.3816),
                    (0.8121, 0.4689, -0.3816),
                    (-0.8121, 0.4689, -0.3816)]),
    'H2O': ('OH2', [(0.0000, 0.0000, 0.1173),
                    (0.0000, 0.7572, -0.4692),
                    (0.0000, -0.7572, -0.4692)]),
    'C2H2': ('C2H2', [(0.0000, 0.0000, 0.6013),
                      (0.0000, 0.0000, -0.6013),
                      (0.0000, 0.0000, 1.6644),
                      (0.0000, 0.0000, -1.6644)]),
    'C2H4': ('C2H4', [(0.0000, 0.0000, 0.6695),
                      (0.0000, 0.0000, -0.6695),
                      (0.0000, 0.9289, 1.2321),
                      (0.0000, -0.9289, 1.2321),
                      (0.0000, 0.9289, -1.2321),
                      (0.0000, -0.9289, -1.2321)]),
    'HCN': ('CHN', [(0.0000, 0.0000, 0.0000),
                    (0.0000, 0.0000, 1.0640),
                    (0.0000, 0.0000, -1.1560)]),
    'Be2': ('Be2', [(0.0000, 0.0000, 0.0000),
                    (0.0000, 0.0000, 2.460)])}


c = ase.db.connect('results.db')

for name in list(ex_atomization.keys()) + 'H Li Be B C N O F Cl P'.split():
    print(name)
    id = c.reserve(name=name)
    if id is None:
        continue

    if name in extra:
        a = Atoms(*extra[name])
    else:
        a = molecule(name)
        if name in bondlengths:
            a.set_distance(0, 1, bondlengths[name])
    a.cell = [11, 12, 13]
    a.center()

    a.calc = GPAW(xc='PBE',
                  mode=PW(500),
                  txt=name + '.txt')
    a.get_potential_energy()

    e = non_self_consistent_energy(a.calc, 'EXX')
    eexx = e.sum()

    c.write(a, name=name, exx=eexx)
