import pytest

from gpaw import GPAW
from gpaw.dft import MonkhorstPack
from gpaw.bztools import find_high_symmetry_monkhorst_pack


def test_high_symmetry_monkhorst_pack_output(gpw_files):
    gpwname = gpw_files['si_pw']

    kpts1_kc = find_high_symmetry_monkhorst_pack(gpwname, 6.0)

    # kpts returned from this function are not inside the (-0.5, 0.5] interval.
    # Putting them inside the interval..
    kpts1_kc = kpts1_kc - (kpts1_kc > 0.5) + (- kpts1_kc >= 0.5)

    kpts_dict = find_high_symmetry_monkhorst_pack(gpwname, 6.0,
                                                  return_as_dict=True)
    atoms = GPAW(gpwname).atoms
    kpts2_kc = MonkhorstPack(**kpts_dict).build(atoms).kpt_Kc

    assert kpts1_kc == pytest.approx(kpts2_kc, abs=1e-10)
