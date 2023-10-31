from ase.io import read
from gpaw import GPAW, LCAO
from gpaw.mom import prepare_mom_calculation
from gpaw.directmin.tools import excite
from gpaw.directmin.etdm_lcao import LCAOETDM
from gpaw.directmin.derivatives import Davidson

calc = GPAW(xc='PBE',
            mode=LCAO(),
            h=0.2,
            basis='dzp',
            spinpol=True,
            eigensolver={'name': 'etdm-lcao', 'representation': 'u-invar'},
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

# Stability analysis using the generalized Davidson method
davidson = Davidson(
    calc.wfs.eigensolver, 'davidson_tPP_constrained.txt', eps=1e-2, seed=42)
appr_sp_order = davidson.estimate_sp_order(
    calc, method='full-hess', target_more=3)
print(appr_sp_order)
