from gpaw import GPAW, setup_paths
from gpaw.xas import RecursionMethod

setup_paths.insert(0, '.')

name = 'diamond333_hch'

calc = GPAW(name + '.gpw',
            kpts=(6, 6, 6),
            txt=name + '_rec.txt',
            legacy_gpaw=True)
calc.initialize()
calc.set_positions()

r = RecursionMethod(calc)
r.run(600)

r.run(1400,
      inverse_overlap='approximate')

r.write(name + '_600_1400a.rec',
        mode='all')
