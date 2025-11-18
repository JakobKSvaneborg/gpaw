from gpaw import GPAW, PW, FermiDirac
from ase.build import mx2

slab = mx2('MoS2', a=3.1841, thickness=3.1271, vacuum=7.5)

name_calc = 'calc_MoS2_BSEPlus'
calc_bse = 'fixed_density_calc_MoS2_bse'
calc_rpa = 'fixed_density_calc_MoS2_rpa'

calc = GPAW(mode=PW(800),
            xc='PBE',
            nbands=200,
            convergence={'bands': -5, 'density': 0.0001,
                         'eigenstates': 4e-08, 'energy': 0.0005},
            occupations=FermiDirac(width=0.01),
            kpts={'density': 26, 'gamma': True},
            txt='gs_MoS2.txt')

slab.calc = calc
slab.get_potential_energy()
calc.write(name_calc + ".gpw")

calc_es = GPAW(name_calc + '.gpw',
               txt='bse_calc.txt',
               fixdensity=True,
               convergence={'bands': 50},
               nbands=100, parallel={'domain': 1},
               kpts={'density': 11.7, 'gamma': True})

slab.calc = calc_es
slab.get_potential_energy()
calc_es.diagonalize_full_hamiltonian(nbands=100)
calc_es.write(calc_bse + '.gpw', mode='all')

calc_es = GPAW(name_calc + '.gpw',
               txt='rpa_calc.txt',
               fixdensity=True,
               convergence={'bands': 80, 'density': 0.00001,
                            'eigenstates': 4e-08, 'energy': 0.00005},
               nbands=220,
               kpts={'density': 23.5, 'gamma': True})

slab.calc = calc_es
slab.get_potential_energy()
calc_es.write(calc_rpa + '.gpw', mode='all')
