from ase.io import read
from gpaw import GPAW, LCAO
from gpaw.directmin.etdm_lcao import LCAOETDM
from gpaw.directmin.tools import excite

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
            txt='N-Phenylpyrrole_GS.txt')

atoms = read('N-Phenylpyrrole.xyz')
atoms.center(vacuum=5.0)
atoms.set_pbc(False)
atoms.calc = calc

# Ground state calculation
E_GS = atoms.get_potential_energy()

# Spin-mixed open-shell occupation numbers
f = excite(calc, 0, 0, spin=(0, 0))

# Direct approach using ground state orbitals with changed occupation numbers
calc.set(eigensolver=LCAOETDM(searchdir_algo={'name': 'l-bfgs-p_gmf'},
                              linesearch_algo={'name': 'max-step'},
                              partial_diagonalizer={
                                  'name': 'Davidson',
                                  'logfile': 'davidson_tPP.txt',
                                  'sp_order': 6,
                                  'seed': 42},
                              update_ref_orbs_counter=1000,
                              representation='u-invar',
                              need_init_orbs=False),
         occupations={'name': 'mom', 'numbers': f,
                      'use_fixed_occupations': True},
         txt='N-Phenylpyrrole_EX_DO-GMF.txt')

E_EX = atoms.get_potential_energy()
