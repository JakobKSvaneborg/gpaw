import pytest
from ase import Atoms
from ase.units import Bohr, Ha

from gpaw import GPAW, FermiDirac
from gpaw.utilities.adjust_cell import adjust_cell
from gpaw.poisson import PoissonSolver
from gpaw.utilities.ewald import madelung


def not_test_fminus(in_tmp_dir):
    atoms = Atoms('F')
    h = 0.2
    adjust_cell(atoms, 2, h=h)
    atoms.pbc = True
    atoms.calc = GPAW(
        mode='fd', h=h, charge=-1,
        occupations=FermiDirac(width=0.1),
        poissonsolver=PoissonSolver(
            use_charge_center=True, use_charged_periodic_corrections=True)
    )
    E = atoms.get_potential_energy()

    assert E > -3


h = 0.3


@pytest.fixture
def mg():
    atoms = Atoms('Mg')
    adjust_cell(atoms, 3, h=h)
    atoms.pbc = True
    return atoms


@pytest.fixture
def mg_plus_uncorrected(in_tmp_dir, mg):
    atoms = mg
    atoms.calc = GPAW(
        mode='fd', h=h, charge=1,
        occupations=FermiDirac(width=0.1),
        txt='no_cpc.out'
    )
    atoms.get_potential_energy()
    return atoms


def test_mgplus_makov1(in_tmp_dir, mg_plus_uncorrected):
    atoms = mg_plus_uncorrected
    E0 = atoms.get_potential_energy()

    atoms.calc.set(
        poissonsolver=PoissonSolver(use_charged_periodic_corrections=True))
    E = atoms.get_potential_energy()

    # the difference should be the first term in the correction
    # by Makov and Payne doi: 10.1103/PhysRevB.51.4014
    assert E - E0 == pytest.approx(madelung(atoms.cell / Bohr) * Ha / 2)
