import pytest

from gpaw import GPAW, LCAO

from ase import Atoms
import numpy as np


@pytest.mark.do
def test_constraints_directopt_lcao(in_tmp_dir):
    # H2O molecule:
    d = 0.9575
    t = np.pi / 180 * 104.51
    H2O = Atoms('OH2',
                positions=[(0, 0, 0),
                           (d, 0, 0),
                           (d * np.cos(t), d * np.sin(t), 0)])
    H2O.center(vacuum=4.0)

    calc = GPAW(mode=LCAO(),
                basis='dzp',
                h=0.22,
                occupations={'name': 'fixed-uniform'},
                eigensolver={'name': 'etdm-lcao'},
                mixer={'backend': 'no-mixing'},
                nbands='nao',
                symmetry='off',
                spinpol=True,
                convergence={'density': 1.0e-4,
                             'eigenstates': 4.0e-8})
    H2O.calc = calc
    H2O.get_potential_energy()

    homo = 3
    lumo = 4
    a = 0.5 * np.pi
    c = calc.wfs.kpt_u[0].C_nM.copy()
    calc.wfs.kpt_u[0].C_nM[homo] = np.cos(a) * c[homo] + np.sin(a) * c[lumo]
    calc.wfs.kpt_u[0].C_nM[lumo] = np.cos(a) * c[lumo] - np.sin(a) * c[homo]

    calc.set(eigensolver={'name': 'etdm-lcao',
                          'constraints': [[[homo], [lumo]], []],
                          'need_init_orbs': False})

    e = H2O.get_potential_energy()

    assert e == pytest.approx(-4.843094, abs=1.0e-4)
