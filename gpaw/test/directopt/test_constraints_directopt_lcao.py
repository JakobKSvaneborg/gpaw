import numpy as np
import pytest

from gpaw import GPAW


@pytest.mark.old_gpaw_only
@pytest.mark.do
def test_constraints_directopt_lcao(in_tmp_dir, gpw_files):
    calc = GPAW(gpw_files['h2o_cdo_lcao'])
    H2O = calc.atoms
    H2O.calc = calc
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
