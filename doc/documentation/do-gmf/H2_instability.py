from gpaw import GPAW, LCAO
from gpaw.directmin.derivatives import Davidson
from ase import Atoms

calc = GPAW(xc='PBE',
            mode=LCAO(),
            h=0.2,
            basis='dzp',
            spinpol=True,
            eigensolver='etdm-lcao',
            occupations={'name': 'fixed-uniform'},
            mixer={'backend': 'no-mixing'},
            nbands='nao',
            symmetry='off',
            txt='H2_GS.txt')

atoms = Atoms('H2', positions=[(0, 0, 0), (0, 0, 2.0)])
atoms.center(vacuum=5.0)
atoms.set_pbc(False)
atoms.calc = calc

# Ground state calculation
E_GS_spin_symmetric = atoms.get_potential_energy()

# Stability analysis using the generalized Davidson method
davidson = Davidson(calc.wfs.eigensolver, 'davidson_H2_S.txt', seed=42)
davidson.run(calc.wfs, calc.hamiltonian, calc.density)

# Break the instability by displacing along the eigenvector of the electronic
# Hessian corresponding to the negative eigenvalue
C_ref = [calc.wfs.kpt_u[x].C_nM.copy() for x in range(len(calc.wfs.kpt_u))]
davidson.break_instability(calc.wfs, n_dim=[10, 10], c_ref=C_ref, number=1)

# Reconverge the electronic structure
calc.calculate(properties=['energy'], system_changes=['positions'])
E_GS_broken_spin_symmetry = atoms.get_potential_energy()

# Repeat stability analysis to confirm that a minimum was found
davidson = Davidson(calc.wfs.eigensolver, 'davidson_H2_BS.txt', seed=42)
davidson.run(calc.wfs, calc.hamiltonian, calc.density)
