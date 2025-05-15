import pytest

from gpaw import GPAW
from gpaw.directmin.derivatives import Davidson
from ase import Atoms


@pytest.mark.old_gpaw_only
@pytest.mark.do
def test_break_instability_lcao(in_tmp_dir, gpw_files):
    calc = GPAW(gpw_files['h2_break_ilcao'])
    # XXX(rg): Remove hack after tchem-gl-13
    calc.set_positions()
    calc.wfs.eigensolver.initialize_dm_helper(calc.wfs, calc.hamiltonian, calc.density, calc.log)
    atoms = calc.atoms
    atoms.calc = calc
    e_symm = atoms.get_potential_energy()
    assert e_symm == pytest.approx(-2.035632, abs=1.0e-3)

    davidson = Davidson(calc.wfs.eigensolver, None, seed=42)
    davidson.run(calc.wfs, calc.hamiltonian, calc.density)

    # Break the instability by displacing along the eigenvector of the
    # electronic Hessian corresponding to the negative eigenvalue
    C_ref = [calc.wfs.kpt_u[x].C_nM.copy()
             for x in range(len(calc.wfs.kpt_u))]
    davidson.break_instability(calc.wfs, n_dim=[10, 10],
                               c_ref=C_ref, number=1)

    calc.calculate(properties=['energy'], system_changes=['positions'])
    e_bsymm = atoms.get_potential_energy()
    assert e_bsymm == pytest.approx(-2.418488, abs=1.0e-3)
