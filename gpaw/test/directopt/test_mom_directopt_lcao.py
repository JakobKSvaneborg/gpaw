import pytest

from gpaw import GPAW, LCAO
from gpaw.directmin.tools import excite
from gpaw.mom import prepare_mom_calculation
from gpaw.directmin.etdm_lcao import LCAOETDM
from gpaw.directmin.tools import rotate_orbitals

from ase import Atoms
import numpy as np


@pytest.mark.mom
def test_mom_directopt_lcao(in_tmp_dir):
    # Water molecule:
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
                eigensolver='etdm-lcao',
                mixer={'backend': 'no-mixing'},
                nbands='nao',
                symmetry='off',
                spinpol=True,
                convergence={'density': 1.0e-4,
                             'eigenstates': 4.0e-8})
    H2O.calc = calc
    H2O.get_potential_energy()

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
