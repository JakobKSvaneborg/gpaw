from ase.build import mx2
from gpaw.new.ase_interface import GPAW

a12 = mx2(formula='MoS2', kind='2H', a=3.184, thickness=3.13,
          size=(1, 1, 1))
a12 += mx2(formula='WS2', kind='2H', a=3.184, thickness=3.15,
           size=(1, 1, 1))
a12.positions[3:, 2] += 3.6 + (3.13 + 3.15) / 2
a12.positions[3:] += [-1 / 3, 1 / 3, 0] @ a12.cell
a12.center(vacuum=6.0, axis=2)
k = 6
a12.calc = GPAW(mode='lcao',
                basis='dzp',
                nbands='nao',
                kpts=dict(size=(k, k, 1), gamma=True),
                eigensolver={'name': 'scissors',
                             'shifts': ([(-0.5, 0.5, 3),
                                         (-0.3, 0.3, 3)])},
                txt='mos2ws2.txt')
a12.get_potential_energy()
bp = a12.cell.bandpath('GMKG', npoints=50)
bscalc = a12.calc.fixed_density(kpts=bp, symmetry='off')
bscalc.write('mos2ws2.gpw')
bs = bscalc.band_structure()
bs.write('mos2ws2.json')
