import pytest
from ase import Atoms
from ase.io import read

from gpaw import GPAW, FermiDirac


@pytest.mark.ci
def test_no_cell():
    with pytest.raises(ValueError):
        H = Atoms('H', calculator=GPAW(mode='fd'))
        H.get_potential_energy()


@pytest.mark.later
def test_read_txt(in_tmp_dir, needs_ase_master):
    a = 2.0
    calc = GPAW(
        mode='fd',
        gpts=(12, 12, 12),
        txt='H.txt',
        occupations=FermiDirac(0.0))
    H = Atoms('H',
              cell=(a, a, a),
              pbc=True,
              calculator=calc)
    e0 = H.get_potential_energy()

    H = read('H.txt')
    assert H.get_potential_energy() == pytest.approx(e0)

    energy_tolerance = 0.001
    assert e0 == pytest.approx(-6.5577, abs=energy_tolerance)
