from gpaw import GPAW, PW, FermiDirac
from ase.build import mx2

slab = mx2('MoS2', kind='2H', a=3.1841, thickness=3.1271, vacuum=7.5)

name_calc = 'calc_MoS2_BSEPlus'
calc_bse = 'fixed_density_calc_MoS2_bse'
calc_rpa = 'fixed_density_calc_MoS2_rpa'

calc = GPAW(mode=PW(800),
            xc='PBE',
            nbands=200,
            convergence={'bands': -5, 'density': 1e-4,
                         'eigenstates': 4e-08, 'energy': 5e-5},
            occupations=FermiDirac(width=0.01),
            kpts={'density': 26, 'gamma': True},
            txt='gs_MoS2.txt')

slab.calc = calc
slab.get_potential_energy()
calc.write(name_calc + ".gpw")

calc_es = GPAW(name_calc + '.gpw').fixed_density(
    txt='bse_calc.txt',
    convergence={'bands': 50},
    nbands=100, parallel={'domain': 1},
    kpts={'density': 11.7, 'gamma': True})
calc_es.diagonalize_full_hamiltonian(nbands=100)
calc_es.write(calc_bse + '.gpw', mode='all')

calc_es = GPAW(name_calc + '.gpw').fixed_density(
    txt='rpa_calc.txt',
    convergence={'bands': 80, 'density': 1e-5,
                 'eigenstates': 4e-8, 'energy': 5e-5},
    nbands=220,
    kpts={'density': 23.5, 'gamma': True})
calc_es.write(calc_rpa + '.gpw', mode='all')
