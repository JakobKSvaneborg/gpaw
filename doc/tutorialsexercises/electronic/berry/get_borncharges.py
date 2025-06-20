# web-page: born_charges_BaTiO3.csv
from ase.io.jsonio import read_json
import numpy as np

jsn_file = 'born_charges.json'

results = read_json(jsn_file)
Z_avv = results['Z_avv']
sym_a = results['sym_a']

for sym, Z_vv in zip(sym_a, Z_avv):
    print(sym)
    print(np.round(Z_vv, 2))

# --- literalinclude import-end ---
csv_file = 'born_charges_BaTiO3.csv'


def write_csv(csv_file, Z_avv, sym_a):
    from ase.parallel import paropen

    with paropen(csv_file, 'w') as fd:
        for sym, Z_vv in zip(sym_a, Z_avv):
            for row in Z_vv:
                fd.write(' ,{:4.2f},{:4.2f},{:4.2f}\n'.format(*row))


write_csv(csv_file, Z_avv, sym_a)
