# web-page: born_charges_BaTiO3.csv
from ase.io.jsonio import read_json
import numpy as np
from ase.parallel import parprint, paropen

jsn_file = 'born_charges.json'
csv_file = 'born_charges_BaTiO3.csv'

results = read_json(jsn_file)
Z_avv = results['Z_avv']
sym_a = results['sym_a']

with paropen(csv_file, 'w') as fd:
    for sym, Z_vv in zip(sym_a, Z_avv):
        parprint(sym)
        parprint(np.round(Z_vv, 2))
        fd.write(f'{sym}, , , \n')
        for row in Z_vv:
            fd.write(' ,{:6.4f},{:6.4f},{:6.4f}\n'.format(*row))
