from ase.build import bulk

from gpaw import GPAW, FermiDirac
from gpaw import PW
from gpaw.response.g0w0 import G0W0

a = 3.567
atoms = bulk('C', 'diamond', a=a)

calc = GPAW(mode=PW(600),
            parallel={'domain': 1},
            kpts={'size': (4, 4, 4), 'gamma': True},
            xc='LDA',
            occupations=FermiDirac(0.001),
            txt='C_groundstate_freq2.txt')

atoms.calc = calc
atoms.get_potential_energy()

calc.diagonalize_full_hamiltonian()
calc.write('C_groundstate_freq2.gpw', 'all')

for i, domega0 in enumerate([0.01, 0.02, 0.025, 0.03, 0.04, 0.05]):
    for j, omega2 in enumerate([1, 5, 10, 15, 20, 25]):
        frequencies = {'type': 'nonlinear',
                       'domega0': domega0,
                       'omega2': omega2}
        gw = G0W0(calc='C_groundstate_freq2.gpw',
                  # nbands=30,
                  bands=(3, 5),
                  kpts=[0],
                  ecut=200,
                  integrate_gamma='WS',
                  frequencies=frequencies,
                  filename=f'C_g0w0_domega0_{domega0}_omega2_{omega2}_2')

        results = gw.calculate()
