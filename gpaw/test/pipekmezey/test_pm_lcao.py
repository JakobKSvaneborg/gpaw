import pytest
from ase import Atoms

from gpaw import GPAW
from gpaw.wannier.pipekmezey.pipek_mezey_wannier import PipekMezey


@pytest.mark.pipekmezey
def test_pipekmezey_lcao_hirshfeld(in_tmp_dir):
    atoms = Atoms('CO',
                  positions=[[0, 0, 0],
                             [0, 0, 1.128]])
    atoms.center(vacuum=5)
    calc = GPAW(mode='lcao',
                h=0.24,
                convergence={'density': 1e-4})
    atoms.calc = calc
    atoms.get_potential_energy()
    PM = PipekMezey(calc=calc,
                    method='H',
                    seed=42)
    PM.localize()
    P = PM.get_function_value()
    assert P == pytest.approx(3.3263, abs=0.0001)
    PM = PipekMezey(calc=calc,
                    seed=42)
    PM.localize()
    P = PM.get_function_value()
    assert P == pytest.approx(3.3756, abs=0.002)
