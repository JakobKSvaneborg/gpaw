import numpy as np
from ase import Atoms
from ase.build import molecule

from gpaw import GPAW

L = 3.0 + 2 * 4.0
d = np.linspace(3.0, 5.5, 11)
for symbol in ['Ar', 'Kr']:
    dimer = Atoms([symbol, symbol],
                  [(0, 0, 0), (1, 1, 1)],
                  cell=(L, L, L))
    dimer.center()
    calc = GPAW(mode='fd', h=0.2, xc='revPBE', txt=symbol + '-dimer.txt')
    dimer.calc = calc
    for r in d:
        dimer.set_distance(0, 1, r)
        e = dimer.get_potential_energy()
        calc.write('%s-dimer-%.2f.gpw' % (symbol, r))
    del dimer[1]
    dimer.calc = calc.new(txt=symbol + '-atom.txt')
    e = dimer.get_potential_energy()
    dimer.calc.write('%s-atom.gpw' % symbol)

dimer = molecule('C6H6')
dimer += dimer
dimer.positions[12:, 2] += 5.5
dimer.center(vacuum=4.0)
calc = GPAW(mode='fd', h=0.2, xc='revPBE', txt='benzene-dimer.txt')
dimer.calc = calc
for r in d:
    dimer.positions[12:, 2] = dimer.positions[:12, 2] + r
    dimer.center()
    e = dimer.get_potential_energy()
    calc.write('benzene-dimer-%.2f.gpw' % r)

del dimer[12:]
dimer.calc = calc.new(txt='benzene.txt')
e = dimer.get_potential_energy()
dimer.calc.write('benzene.gpw')
