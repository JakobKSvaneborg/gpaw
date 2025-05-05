import pytest

from ase.build import molecule
from gpaw.new.ase_interface import GPAW


@pytest.mark.parametrize('mode', ['pw', 'fd'])
@pytest.mark.parametrize('eigensolver', ['dav', 'rmm-diis'])
@pytest.mark.parametrize('max_mem', [-50, 0, 1024**6])
def test_max_buffer_mem(mode, eigensolver, max_mem):
    atoms = molecule('H2', vacuum=1)
    calc = GPAW(mode=mode,
                eigensolver={'name': eigensolver,
                             'max_buffer_mem': max_mem},
                xc='LDA',
                convergence={'maximum iterations': 2})
    atoms.calc = calc
    e = atoms.get_potential_energy()

    expected_e = {'pw-rmm-diis': -14.399880,
                  'fd-rmm-diis': 5.9194033,
                  'pw-dav': -16.0133410,
                  'fd-dav': 4.6795767}
    assert e == pytest.approx(expected_e[f'{mode}-{eigensolver}'], abs=1e-3)
