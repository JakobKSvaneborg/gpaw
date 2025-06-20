from ase.io.jsonio import read_json
import numpy as np

Z_avv = read_json('born_charges.json')['Z_avv']
for a, Z_vv in enumerate(Z_avv):
    print(a)
    print(np.round(Z_vv, 2))
