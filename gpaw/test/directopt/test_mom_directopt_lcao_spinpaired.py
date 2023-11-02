import numpy as np
import pytest
from ase import Atoms
from gpaw import GPAW
from gpaw.mom import prepare_mom_calculation
from gpaw.directmin.etdm_lcao import LCAOETDM
from gpaw.directmin.tools import excite


@pytest.mark.mom
def test_mom_directopt_lcao_spinpaired(in_tmp_dir):
    atoms = Atoms('C2H4',
                  [[6.68748500e-01, 2.00680000e-04, 5.55800000e-05],
                   [-6.68748570e-01, -2.00860000e-04, -5.51500000e-05],
                   [4.48890600e-01, -5.30146300e-01, 9.32670330e-01],
                   [4.48878120e-01, -5.30176640e-01, -9.32674730e-01],
                   [-1.24289513e+00, 1.46164400e-02, 9.32559990e-01],
                   [-1.24286000e+00, -1.46832100e-02, -9.32554970e-01]])
    atoms.center(vacuum=4)

    eigensolver = LCAOETDM(searchdir_algo={'name': 'l-sr1p'},
                           linesearch_algo={'name': 'max-step'})

    calc = GPAW(mode='lcao',
                basis='dzp',
                h=0.24,
                xc='PBE',
                symmetry='off',
                occupations={'name': 'fixed-uniform'},
                eigensolver=eigensolver,
                mixer={'backend': 'no-mixing'},
                nbands='nao',
                convergence={'density': 1.0e-4,
                             'eigenstates': 4.0e-8})
    atoms.calc = calc
    atoms.get_potential_energy()

    f_sn = excite(calc, 0, 0, spin=(0, 0))
    f_sn[0] /= 2

    prepare_mom_calculation(calc, atoms, f_sn)
    # This fails if the memory of the search direction
    # algorithm is not erased
    e = atoms.get_potential_energy()

    calc.wfs.occupations.initialize_reference_orbitals()
    calc.wfs.calculate_occupation_numbers(calc.density.fixed)

    # These fail if the OccupationsMOM.numbers are not updated correctly
    assert np.all(calc.get_occupation_numbers() <= 2.0)
    assert e == pytest.approx(-21.38257404436053, abs=0.01)
