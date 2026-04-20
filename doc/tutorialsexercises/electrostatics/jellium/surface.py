import json
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.units import Bohr
from gpaw import GPAW, PW
from gpaw.core import UGArray
from gpaw.new.extensions import Jellium

rs = 5.0 * Bohr  # Wigner-Seitz radius
h = 0.2          # grid-spacing
a = 8 * h        # lattice constant
v = 3 * a        # vacuum
L = 10 * a       # thickness
k = 12           # number of k-points (k*k*1)

ne = a**2 * L / (4 * np.pi / 3 * rs**3)

eps = 0.001  # displace surfaces away from grid-points


class JelliumSlab(Jellium):
    def update_mask(self, mask_r: UGArray) -> None:
        z = mask_r.desc.xyz()[:, :, :, 2]
        mask_r.data[np.logical_and(z > v - eps, z < v + L + eps)] = 1.0


surf = Atoms(pbc=(True, True, False),
             cell=(a, a, v + L + v))
surf.calc = GPAW(mode=PW(400.0),
                 extensions=[JelliumSlab(charge=ne)],
                 # background_charge=jellium,
                 charge=-ne,
                 xc='LDA_X+LDA_C_WIGNER',
                 kpts=[k, k, 1],
                 h=h,
                 convergence={'density': 0.001},
                 nbands=int(ne / 2) + 15,
                 txt='surface.txt')
e = surf.get_potential_energy()
density = surf.calc.get_pseudo_density()[0, 0]
Path('surface.json').write_text(json.dumps(density.tolist()))
