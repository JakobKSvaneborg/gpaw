from gpaw import GPAW
from gpaw import PW
from gpaw.occupations import FermiDirac
from ase.spacegroup import crystal

aa = 4.584
c = 2.953

a = crystal(['Ti', 'O'], basis=[(0, 0, 0), (0.3, 0.3, 0.0)],
            spacegroup=136, cellpar=[aa, aa, c, 90, 90, 90])

name_calc = 'calc_BSE_plus'
name_bse_plus = 'fixed_density_calc_BSE_Plus'
name_rpa = 'fixed_density_calc_rpa'

calc = GPAW(mode=PW(800),
            xc='PBE',
            occupations=FermiDirac(width=0.01),
            parallel={'domain': 1, 'band': 1},
            convergence={'bands': 50, 'density': 0.0001,
                         'eigenstates': 4e-07, 'energy': 0.0005},
            kpts={'density': 8, 'gamma': True, 'even': True})
a.calc = calc
a.get_potential_energy()
calc.write(name_calc + ".gpw")

calc_es = GPAW(name_calc + '.gpw',
               fixdensity=True,
               convergence={'bands': 50},
               nbands=60, parallel={'domain': 1},
               kpts={'density': 6, 'gamma': True, 'even': True})

a.calc = calc_es
a.get_potential_energy()
calc_es.diagonalize_full_hamiltonian(nbands=100)
calc_es.write(name_bse_plus + '.gpw', mode='all')

calc_es = GPAW(name_calc + '.gpw',
               fixdensity=True,
               convergence={'bands': 100},
               nbands=240, parallel={'domain': 4},
               kpts={'density': 15, 'gamma': True, 'even': True})

a.calc = calc_es
a.get_potential_energy()
calc_es.write(name_rpa + '.gpw', mode='all')
