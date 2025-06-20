from ase.io.jsonio import read_json
import numpy as np

results = read_json('born_charges.json')
Z_avv = results['Z_avv']
sym_a = results['sym_a']

for sym, Z_vv in zip(sym_a, Z_avv):
    print(sym)
    print(np.round(Z_vv, 2))
