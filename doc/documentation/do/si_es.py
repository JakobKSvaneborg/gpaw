from gpaw import GPAW
from ase.build import bulk
from gpaw.directmin.etdm_fdpw import FDPWETDM
from gpaw.mom import prepare_mom_calculation
from gpaw.directmin.tools import excite
from ase.parallel import paropen


atoms = bulk('Si', 'diamond', a=5.44, cubic=True)
atoms.cell[0, 0] += 1e-3

# Step: Set up the GPAW calculator
calc = GPAW(mode={'name': 'pw',    # Use plane wave mode
                  'ecut': 340},   # Cutoff energy
            xc='PBE',
            kpts=(1, 1, 1),
            eigensolver=FDPWETDM(converge_unocc=True),
            mixer={'backend': 'no-mixing'},
            occupations={'name': 'fixed-uniform'},
            spinpol=True,
            )

atoms.calc = calc
E_gs = atoms.get_potential_energy()


calc.set(eigensolver=FDPWETDM(excited_state=True,
                              converge_unocc=False,
                              momevery=10,
                              max_step_inner_loop=0.2,
                              maxiter_inner_loop=20))


f_sn = excite(calc, 0, 0, (0, 0))
prepare_mom_calculation(calc, atoms, f_sn)
E_es = atoms.get_potential_energy()

print(f'Excitation energy: {E_es - E_gs}')

with paropen('si_excited.txt', 'w') as fd:
    print(f'Excitation energy Si: {E_es - E_gs} eV',
          file=fd)
