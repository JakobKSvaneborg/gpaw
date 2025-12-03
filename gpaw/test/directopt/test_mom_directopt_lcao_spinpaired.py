import numpy as np
import pytest

from gpaw import GPAW
from gpaw.directmin.tools import excite
from gpaw.mom import prepare_mom_calculation


@pytest.mark.mom
@pytest.mark.do
def test_mom_directopt_lcao_spinpaired(in_tmp_dir, gpw_files):
    calc = GPAW(gpw_files['c2h4_do_lcao'])
    atoms = calc.atoms
    atoms.calc = calc

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
