from gpaw import GPAW, LCAO
from ase import Atoms
import numpy as np
from gpaw.directmin.etdm_lcao import LCAOETDM
# Water molecule:
d = 0.9575
t = np.pi / 180 * 104.51
H2O = Atoms('OH2',
            positions=[(0, 0, 0),
                       (d, 0, 0),
                       (d * np.cos(t), d * np.sin(t), 0)])
H2O.center(vacuum=5.0)

calc = GPAW(mode=LCAO(),
            basis='dzp',
            eigensolver=LCAOETDM(
                searchdir_algo={'name': 'l-bfgs-p', 'memory': 10}),
            occupations={'name': 'fixed-uniform'},
            mixer={'backend': 'no-mixing'},
            nbands='nao'
            )
H2O.set_calculator(calc)
H2O.get_potential_energy()
