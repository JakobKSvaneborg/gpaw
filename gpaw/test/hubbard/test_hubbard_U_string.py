from ase.units import Hartree

from gpaw.hubbard import parse_hubbard_string


def test_hubbard_u_string():
    # Format of string = type:l,U,scale;l,U,scale
    # U has an internal unit conversion to atomic units

    # Simple single correction
    string = ':d,5,0'
    setuptype, hubbard = parse_hubbard_string(string)
    assert setuptype == 'paw'
    assert hubbard._tuple() == ([2], [5.0 / Hartree], [False])


def test_single_correction_without_scale():
    string = ':d,5'
    setuptype, hubbard = parse_hubbard_string(string)
    assert setuptype == 'paw'
    assert hubbard._tuple() == ([2], [5.0 / Hartree], [True])


def test_multi_correction():
    string = 'ae:s,3,0;p,2.5,1;f,0.5,0'
    expected = ([0, 1, 3],
                [3.0 / Hartree, 2.5 / Hartree, 0.5 / Hartree],
                [False, True, False])
    setuptype, hubbard = parse_hubbard_string(string)
    assert setuptype == 'ae'
    assert hubbard._tuple() == expected
