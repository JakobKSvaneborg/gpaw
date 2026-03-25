from ase.io import read


def test_gpaw_log(gpw_files):
    path = gpw_files['h2_pw'].with_suffix('.txt')
    atoms = read(path, format='gpaw-log')
    assert len(atoms) == 2
