from ase.io import read
from gpaw import GPAW, LCAO
from gpaw.mom import prepare_mom_calculation
from gpaw.directmin.tools import excite
from gpaw.directmin.etdm_lcao import LCAOETDM

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

# Ground state LCAO coefficients and occupation numbers
C_GS = [calc.wfs.kpt_u[x].C_nM.copy() for x in range(len(calc.wfs.kpt_u))]
f_gs = [calc.wfs.kpt_u[x].f_n.copy() for x in range(len(calc.wfs.kpt_u))]

# Direct approach using ground state orbitals with changed occupation numbers
calc.set(eigensolver=LCAOETDM(searchdir_algo={'name': 'l-sr1p'},
                              linesearch_algo={'name': 'max-step'},
                              need_init_orbs=False),
         txt='N-Phenylpyrrole_EX_direct.txt')

# Spin-mixed open-shell occupation numbers
f = excite(calc, 0, 0, spin=(0, 0))

# Direct optimization maximum overlap method calculation
prepare_mom_calculation(calc, atoms, f)
E_EX_direct = atoms.get_potential_energy()

# Reset LCAO coefficients and occupation numbers to ground state solution
for k, kpt in enumerate(calc.wfs.kpt_u):
    kpt.C_nM = C_GS[k]
    kpt.f_n = f_gs[k]

h = 26  # Hole
p = 27  # Excited electron

# Constrained optimization freezing hole and excited electron
calc.set(eigensolver=LCAOETDM(constraints=[[[h], [p]], []],
                              need_init_orbs=False),
         txt='N-Phenylpyrrole_EX_constrained.txt')

# Spin-mixed open-shell occupation numbers
f = excite(calc, 0, 0, spin=(0, 0))

# Direct optimization maximum overlap method calculation
prepare_mom_calculation(calc, atoms, f)
E_EX_constrained = atoms.get_potential_energy()

# Unconstrained optimization using constrained solution as initial guess
calc.set(eigensolver=LCAOETDM(searchdir_algo={'name': 'l-sr1p'},
                              linesearch_algo={'name': 'max-step'},
                              need_init_orbs=False),
         txt='N-Phenylpyrrole_EX_from_constrained.txt')

# Direct optimization maximum overlap method calculation
E_EX_from_constrained = atoms.get_potential_energy()
