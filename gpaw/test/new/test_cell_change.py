import pytest
from ase import Atoms
from gpaw.new.ase_interface import GPAW
from gpaw.mpi import broadcast_string
from io import StringIO


@pytest.mark.parametrize('gpu', [False, True])
def test_new_cell(gpu):
    a = 2.1
    az = 2.099
    atoms = Atoms('Li', pbc=True, cell=[a, a, az])
    atoms.positions += 0.01
    output = StringIO()
    atoms.calc = GPAW(
        xc='PBE',
        mode={'name': 'pw'},
        kpts=(2, 2, 1),
        parallel={'gpu': gpu, 'domain': 1},
        txt=output)
    e0 = atoms.get_potential_energy()
    s0 = atoms.get_stress()
    f0 = atoms.get_forces()
    print(e0, s0, f0)
    assert e0 == pytest.approx(-1.27648045935401)
    assert f0 == pytest.approx(0, abs=1e-5)
    assert s0 == pytest.approx([-3.97491456e-01] * 2
                               + [3.29507807e-03] + [0, 0, 0], abs=5e-7)

    atoms.cell[2, 2] = 0.9 * az
    atoms.positions += 0.1
    e1 = atoms.get_potential_energy()
    s1 = atoms.get_stress()
    f1 = atoms.get_forces()
    print(e1, s1, f1)
    assert e1 == pytest.approx(-1.2359952570422994)
    assert f1 == pytest.approx(0, abs=1e-4)
    assert s1 == pytest.approx([-4.37458548e-01] * 2 +
                               [-9.41665221e-02, 0.0, 0.0, 0.0], abs=5e-7)
    out = broadcast_string(output.getvalue() or None)
    assert '# Interpolating wave fun' in out
