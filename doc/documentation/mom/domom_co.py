from ase.build import molecule
from ase.optimize import BFGS
from ase.parallel import paropen
from gpaw import GPAW, LCAO
from gpaw.mom import prepare_mom_calculation
from gpaw.directmin.tools import excite
from gpaw.directmin.etdm_lcao import LCAOETDM


for spinpol in [True, False]:
    if spinpol:
        tag = 'spinpol'
    else:
        tag = 'spinpaired'

    atoms = molecule('CO')
    atoms.center(vacuum=5)

    calc = GPAW(xc='PBE',
                mode=LCAO(force_complex_dtype=True),
                h=0.2,
                basis='dzp',
                spinpol=spinpol,
                eigensolver='etdm-lcao',
                occupations={'name': 'fixed-uniform'},
                mixer={'backend': 'no-mixing'},
                nbands='nao',
                symmetry='off',
                txt='co_' + tag + '.txt')
    atoms.calc = calc

    # Ground-state calculation
    E_gs = atoms.get_potential_energy()

    # Prepare initial guess for complex pi* orbitals by taking
    # linear combination of real pi*x and pi*y orbitals
    lumo = 5  # lumo is pi*x or pi*y orbital
    for kpt in calc.wfs.kpt_u:
        pp = kpt.C_nM[lumo] + 1.0j * kpt.C_nM[lumo + 1]
        pm = kpt.C_nM[lumo] - 1.0j * kpt.C_nM[lumo + 1]
        kpt.C_nM[lumo][:] = pm
        kpt.C_nM[lumo + 1][:] = pp

    calc.set(eigensolver=LCAOETDM(searchdir_algo={'name': 'l-sr1p'},
                                  linesearch_algo={'name': 'max-step'},
                                  need_init_orbs=False))

    # Occupation numbers for sigma->pi* excited state:
    # Remove one electron from homo (sigma) and add one electron to lumo (pi*)
    f = excite(calc, 0, 0, spin=(0, 0))
    if not spinpol:
        f[0] /= 2

    # Prepare excited-state DO-MOM calculation
    prepare_mom_calculation(calc, atoms, f)

    opt = BFGS(atoms, logfile='co_' + tag + '.log', maxstep=0.05)
    opt.run(fmax=0.05)

    d = atoms.get_distance(0, 1)

    with paropen('co_' + tag + '.log', 'a') as fd:
        print(f'Optimized CO bond length sigma->pi* state: {d:.2f} Å', file=fd)
        # https://doi.org/10.1007/978-1-4757-0961-2
        print('Experimental CO bond length sigma->pi* state: 1.24 Å', file=fd)
