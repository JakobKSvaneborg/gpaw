import pytest

from gpaw import GPAW
import numpy as np


@pytest.mark.old_gpaw_only
@pytest.mark.do
def test_constraints_directopt_lcao_sic(in_tmp_dir, gpw_files):
    """
    test Perdew-Zunger Self-Interaction
    Correction  in LCAO mode using DirectMin
    :param in_tmp_dir:
    :return:
    """
    calc = GPAW(gpw_files['h2o_cdo_lcao_sic'])
    H2O = calc.atoms
    H2O.calc = calc

    test_restart = True
    if test_restart:
        from gpaw import restart
        calc.write('h2o.gpw', mode='all')
        H2O, calc = restart('h2o.gpw', txt='-')
        H2O.calc.results.pop('energy')
        H2O.calc.scf.converged = False
        calc.set(eigensolver={'name': 'etdm-lcao',
                              'functional': {'name': 'PZ-SIC',
                                             'scaling_factor': (0.5, 0.5)},
                              'need_init_orbs': False})
        e = H2O.get_potential_energy()
        niter = calc.get_number_of_iterations()
        assert niter == pytest.approx(3, abs=3)
        assert e == pytest.approx(-12.16353, abs=1.0e-3)

    homo = 3
    lumo = 4
    a = 0.5 * np.pi
    c = calc.wfs.kpt_u[0].C_nM.copy()
    calc.wfs.kpt_u[0].C_nM[homo] = np.cos(a) * c[homo] + np.sin(a) * c[lumo]
    calc.wfs.kpt_u[0].C_nM[lumo] = np.cos(a) * c[lumo] - np.sin(a) * c[homo]

    calc.set(eigensolver={'name': 'etdm-lcao',
                          'functional': {'name': 'PZ-SIC',
                                         'scaling_factor': (0.5, 0.5)},
                          'constraints': [[[homo], [lumo]]],
                          'need_init_orbs': False})

    e = H2O.get_potential_energy()

    assert e == pytest.approx(24.24718, abs=0.1)
