import pytest
from ase import Atoms
from gpaw import GPAW
from gpaw.pipekmezey.pipek_mezey_wannier import PipekMezey
from gpaw.test import equal


@pytest.mark.pipekmezey
def test_pipekmezey_fd(in_tmp_dir):

    atoms = Atoms('CO',
                  positions=[[0, 0, 0],
                             [0, 0, 1.128]])
    atoms.center(vacuum=5)

    calc = GPAW(mode='fd',
                h=0.24,
                convergence={'density': 1e-4,
                             'eigenstates': 1e-4})

    calc.atoms = atoms
    calc.calculate()

    PM = PipekMezey(calc=calc, seed=42)
    PM.localize()

    P = PM.get_function_value()

    equal(P, 3.365, 0.001)
