# web-page: born_charges_BaTiO3.csv
# --- literalinclude import-start ---
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


def write_csv(csv_file, Z_avv, sym_a, tol=1e-2):
    from ase.parallel import paropen

    table = []
    for sym, Z_vv in zip(sym_a, Z_avv):
        subtable = []
        for row in Z_vv:
            trow = ['']
            for cell in row:
                if np.abs(cell) > tol:
                    tcell = f'{cell:4.2f}'
                else:
                    tcell = '0'
                trow.append(tcell)
            subtable.append(trow)
        table.append(subtable)

    table = np.transpose(np.array(table), (1, 0, 2))
    table = table.reshape((3, -1))
    with paropen(csv_file, 'w') as fd:
        for row in table:
            fd.write(','.join(row) + '\n')


write_csv(csv_file, Z_avv, sym_a)
