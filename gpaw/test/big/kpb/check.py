import ase.db
from ase.units import kcal, mol

# flake8: noqa
# atomization energies in kcal / mol (= 43.364 meV)
# All values evaluated with PBE xc-orbitals and densities at
# experimental geometries. Zero-point vibration has been removed
# from experimental energies (from [1]).
atomization = {
    # Molec   expt    LSD    PBE    RPBE   BLYP
    'H2'  : ( 109.5, 113.2, 104.6, 105.5, 109.4),
    'LiH' : (  57.8,  61.0,  53.5,  53.4,  58.1),
    'CH4' : ( 419.3, 462.3, 419.8, 410.6, 416.6),
    'NH3' : ( 297.4, 337.3, 301.7, 293.2, 301.4),
    'OH'  : ( 106.4, 124.1, 109.8, 106.3, 109.6),
    'H2O' : ( 232.2, 266.5, 234.2, 226.6, 232.5),
    'HF'  : ( 140.8, 162.2, 142.0, 137.5, 141.0),
    'Li2' : (  24.4,  23.9,  19.9,  20.2,  20.5),
    'LiF' : ( 138.9, 156.1, 138.6, 132.9, 140.1),
    'Be2' : (   3.0,  12.8,   9.8,   7.9,   6.1),
    'C2H2': ( 405.4, 460.3, 414.9, 400.4, 405.3),
    'C2H4': ( 562.6, 632.6, 571.5, 554.5, 560.7),
    'HCN' : ( 311.9, 361.0, 326.1, 313.6, 320.3),
    'CO'  : ( 259.3, 299.1, 268.8, 257.9, 261.8),
    'N2'  : ( 228.5, 267.4, 243.2, 232.7, 239.8),
    'NO'  : ( 152.9, 198.7, 171.9, 161.6, 166.0),
    'O2'  : ( 120.5, 175.0, 143.7, 133.3, 135.3),
    'F2'  : (  38.5,  78.2,  53.4,  45.6,  49.4),
    'P2'  : ( 117.3, 143.8, 121.1, 114.1, 121.0),
    'Cl2' : (  58.0,  83.0,  65.1,  58.9,  57.2)}

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


c = ase.db.connect('results.db')

# Energy of atoms:
atoms = {}
for d in c.select(natoms=1):
    atoms[d.numbers[0]] = [d.energy, d.exx]

maepbe = 0.0
maeexx = 0.0
print('                 PBE                   EXX')
print(('-' * 48))
for d in c.select('natoms>1'):
    epberef = atomization[d.name][2] * kcal / mol
    eexxref = ex_atomization[d.name][0] * kcal / mol

    epbe = sum(atoms[atom][0] for atom in d.numbers) - d.energy
    eexx = sum(atoms[atom][1] for atom in d.numbers) - d.exx

    maepbe += abs(epbe - epberef) / len(ex_atomization)
    maeexx += abs(eexx - eexxref) / len(ex_atomization)

    print(('%-4s %10.3f %10.3f %10.3f %10.3f' %
          (d.name, epbe, epbe - epberef, eexx, eexx - eexxref)))

print(('-' * 48))
print(('MAE  %10.3f %10.3f %10.3f %10.3f' % (0, maepbe, 0, maeexx)))

assert maepbe < 0.025
assert maeexx < 0.05
