import pytest
from gpaw.new.ase_interface import GPAW


def test_afm_h_chain(in_tmp_dir, gpw_files):
    """Compare 2*H AFM cell with 1*H q=1/2 spin-spiral cell."""
    h_calc = GPAW(gpw_files['h_chain'])
    h = h_calc.atoms
    h.calc = h_calc
    e1 = h.get_potential_energy()
    h1, l1 = h.calc.get_homo_lumo()
    # print(e1, h.get_magnetic_moment())
    # print(a1.get_potential_energy(), a1.calc.dft.magmoms())

    h2_calc = GPAW(gpw_files['h2_chain'])
    h2 = h2_calc.atoms
    h2.calc = h2_calc
    e2 = h2.get_potential_energy()
    h2, l2 = h2.calc.get_homo_lumo()

    assert 2 * e1 == pytest.approx(e2, abs=0.002)
    assert h1 == pytest.approx(h2, abs=0.001)
    assert l1 == pytest.approx(l2, abs=0.001)
