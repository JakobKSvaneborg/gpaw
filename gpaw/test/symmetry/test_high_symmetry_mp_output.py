import pytest
from gpaw import GPAW
# from gpaw.bztools import find_high_symmetry_monkhorst_pack
from ase import Atoms

@pytest.mark.xfail
def test_high_symmetry_monkhorst_pack_output(gpw_files):
    atoms = Atoms("H", cell=[6, 3, 8], pbc=True)
    kpts_dict = find_high_symmetry_monkhorst_pack(atoms, 3.0)
    assert kpts_dict["size"] == (4, 8, 4)

    for system, kpts in [("si_pw", (16, 16, 16)), ("mos2_pw", (18, 18, 1))]:
        gpwname = gpw_files[system]
        atoms = GPAW(gpwname).atoms
        kpts_dict = find_high_symmetry_monkhorst_pack(atoms, 6.0)
        assert kpts_dict["size"] == kpts

    for system, kpts in [("si_pw", (8, 8, 8)), ("mos2_pw", (12, 12, 1))]:
        gpwname = gpw_files[system]
        atoms = GPAW(gpwname).atoms
        kpts_dict = find_high_symmetry_monkhorst_pack(atoms, 3.0)
        assert kpts_dict["size"] == kpts
