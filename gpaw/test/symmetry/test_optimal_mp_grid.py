from gpaw import GPAW
from gpaw.bztools import predicated_monkhorst_pack_grid
from ase import Atoms


def test_predicated_monkhorst_pack_grid(gpw_files):
    atoms = Atoms("H", cell=[6, 3, 8], pbc=True)

    kpts_dict = predicated_monkhorst_pack_grid(atoms, 3.0)
    assert (kpts_dict['size'] == [4, 7, 3]).all()
    assert kpts_dict['gamma'] is False

    kpts_dict = predicated_monkhorst_pack_grid(atoms, 3.0,
                                               contains_ibz_vertices=True,
                                               nmaxperdim=2)
    assert (kpts_dict['size'] == [4, 8, 4]).all()
    assert kpts_dict['gamma'] is True

    # We need a better (not cubic) system for checking is_symmetric_mp_grid..
    kpts_dict = predicated_monkhorst_pack_grid(atoms, 3.0,
                                               is_symmetric_mp_grid=True,
                                               nmaxperdim=2)
    assert (kpts_dict['size'] == [4, 7, 3]).all()
    assert kpts_dict['gamma'] is False

    for system, kpts in [('si_pw', (16, 16, 16)), ('mos2_pw', (18, 18, 1))]:
        atoms = GPAW(gpw_files[system]).atoms
        kpts_dict = predicated_monkhorst_pack_grid(atoms, 8.0,
                                                   contains_ibz_vertices=True,
                                                   nmaxperdim=2)
        assert (kpts_dict['size'] == kpts).all()

    for system, kpts in [('si_pw', (8, 8, 8)), ('mos2_pw', (12, 12, 1))]:
        atoms = GPAW(gpw_files[system]).atoms
        print(system)
        kpts_dict = predicated_monkhorst_pack_grid(atoms, 4.0,
                                                   contains_ibz_vertices=True)
        assert (kpts_dict['size'] == kpts).all()
