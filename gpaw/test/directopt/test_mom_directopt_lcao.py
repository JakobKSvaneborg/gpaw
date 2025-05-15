import pytest

from gpaw import GPAW, LCAO
from gpaw.directmin.tools import excite
from gpaw.mom import prepare_mom_calculation
from gpaw.directmin.etdm_lcao import LCAOETDM
from gpaw.directmin.tools import rotate_orbitals

from ase import Atoms
import numpy as np


@pytest.mark.mom
@pytest.mark.do
def test_mom_directopt_lcao(in_tmp_dir, gpw_files):
    calc = GPAW(gpw_files['h2o_mom_do_lcao'])
    # XXX(rg): Remove hack after tchem-gl-13
    calc.set_positions()
    calc.wfs.eigensolver.initialize_dm_helper(calc.wfs, calc.hamiltonian, calc.density, calc.log)
    H2O = calc.atoms
    H2O.calc = calc

    calc.set(eigensolver=LCAOETDM(excited_state=True))
    f_sn = excite(calc, 0, 0, spin=(0, 0))
    prepare_mom_calculation(calc, H2O, f_sn)

    def rotate_homo_lumo(calc=calc):
        angle = 70
        iters = calc.get_number_of_iterations()
        if iters == 3:
            # Exercise rotate_orbitals
            C_M_old = calc.wfs.kpt_u[0].C_nM.copy()
            rotate_orbitals(calc.wfs.eigensolver, calc.wfs,
                            [[3, 4]], [angle], [0])
            angle *= np.pi / 180.0
            C_M_new = np.cos(angle) * C_M_old[3] + np.sin(angle) * C_M_old[4]
            assert calc.wfs.kpt_u[0].C_nM[3] == \
                   pytest.approx(C_M_new, abs=1e-4)

            counter = calc.wfs.eigensolver.update_ref_orbs_counter
            calc.wfs.eigensolver.update_ref_orbs_counter = iters + 1
            calc.wfs.eigensolver.update_ref_orbitals(calc.wfs,
                                                     calc.hamiltonian,
                                                     calc.density)
            calc.wfs.eigensolver.update_ref_orbs_counter = counter

    calc.attach(rotate_homo_lumo, 1)
    e = H2O.get_potential_energy()

    assert e == pytest.approx(-4.854496813259008, abs=1.0e-4)
