from gpaw import GPAW, PW, FermiDirac, restart
from ase import Atoms
import numpy as np
from gpaw.directmin.etdm_fdpw import FDPWETDM
from gpaw.mom import prepare_mom_calculation
from gpaw.directmin.tools import excite
from ase.io import read
from ase.io import write



# Define lattice vectors
lattice_vectors = np.array([
   	[5.4306998253,         0.0000000000,         0.0000000000],
        [0.0000000000,         5.4306998253,         0.0000000000],
        [0.0000000000,         0.0000000000,         5.4306998253]])



# Define positions of 8 si in the conventional cell
positions = [
     [0.000000000,         0.000000000,         0.000000000],
     [0.000000000,         2.715349913,         2.715349913],
     [2.715349913,         2.715349913,         0.000000000],
     [2.715349913,         0.000000000,         2.715349913],
     [4.073024869,         1.357674956,         4.073024869],
     [1.357674956,         1.357674956,         1.357674956],
     [1.357674956,         4.073024869,         4.073024869],
     [4.073024869,         4.073024869,         1.357674956]]



atoms = Atoms('Si8', positions = positions,
                     cell = lattice_vectors,
                     pbc  = True)


# Step: Set up the GPAW calculator
calc = GPAW(mode= {'name':'pw',   # Use plane wave mode
                   'ecut': 340},  # Cutoff energy
            xc='PBE',
            kpts=(1, 1, 1),
            eigensolver=FDPWETDM(converge_unocc=True),
            mixer={'backend': 'no-mixing'},
            occupations={'name': 'fixed-uniform'},
	    spinpol=True,
            )

atoms.calc = calc
atoms.get_potential_energy()
calc.write('si.gs.gpw', mode='all')



calc.set(eigensolver=FDPWETDM(excited_state=True,
                              converge_unocc=False,
                              momevery=10,
                              max_step_inner_loop= 0.2,
                              maxiter_inner_loop= 20))


f_sn = excite(calc, 0, 0, (0, 0))
prepare_mom_calculation(calc, atoms, f_sn)
atoms.get_potential_energy()
calc.write('si.es.gpw', mode='all')


