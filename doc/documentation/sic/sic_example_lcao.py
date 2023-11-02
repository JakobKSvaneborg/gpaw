import numpy as np
from ase import Atoms
from gpaw import GPAW, LCAO
from gpaw.directmin.etdm_lcao import LCAOETDM

# Water molecule:
d = 0.9575
t = np.pi / 180 * 104.51
H2O = Atoms('OH2',
            positions=[(0, 0, 0),
                       (d, 0, 0),
                       (d * np.cos(t), d * np.sin(t), 0)])
H2O.center(vacuum=5.0)

calc = GPAW(mode=LCAO(force_complex_dtype=True),
            xc='PBE',
            occupations={'name': 'fixed-uniform'},
            eigensolver=LCAOETDM(localizationtype='PM_PZ',
                                 functional={'name': 'PZ-SIC',
                                             'scaling_factor':
                                                 (0.5, 0.5)}),
            mixer={'backend': 'no-mixing'},
            nbands='nao',
            symmetry='off'
            )

H2O.calc = calc
H2O.get_potential_energy()
H2O.get_forces()
