import pytest

from gpaw import GPAW
from gpaw.directmin.derivatives import Davidson
from ase import Atoms


@pytest.mark.do
def test_break_instability_lcao(in_tmp_dir):
    atoms = Atoms('H2', positions=[(0, 0, 0), (0, 0, 2.0)])
    atoms.center(vacuum=2.0)
    atoms.set_pbc(False)

    calc = GPAW(xc='PBE',
                mode='lcao',
                h=0.24,
                basis='sz(dzp)',
                spinpol=True,
                eigensolver='etdm-lcao',
                convergence={'density': 1.0e-2,
                             'eigenstates': 1.0e-2},
                occupations={'name': 'fixed-uniform'},
                mixer={'backend': 'no-mixing'},
                nbands='nao',
                symmetry='off')
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
