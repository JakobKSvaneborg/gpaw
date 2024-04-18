import pytest
from ase import Atoms
from gpaw import GPAW, FermiDirac
from gpaw.utilities.adjust_cell import adjust_cell


def test_Hubbard_U_Zn():
    h = .35
    box = 3.
    energy_tolerance = 0.0004

    s = Atoms('Zn')
    adjust_cell(s, box, h=h)

    E = {}
    E_U = {}
    for spin in [0, 1]:
        params = dict(h=h,
                      spinpol=spin,
                      mode='lcao',
                      basis='sz(dzp)',
                      parallel=dict(kpt=1),
                      charge=1,
                      occupations=FermiDirac(width=0.1, fixmagmom=spin))
        s.calc = GPAW(**params)
        E[spin] = s.get_potential_energy()
        s.calc = GPAW(**params, setups=':d,3.0,1')
        E_U[spin] = s.get_potential_energy()

    assert E[0] == pytest.approx(E[1], abs=energy_tolerance)
    assert E_U[0] == pytest.approx(E_U[1], abs=energy_tolerance)
    assert E_U[0] - E[0] == pytest.approx(-0.167, abs=0.002)
