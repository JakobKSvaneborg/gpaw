from ase.build import bulk
from ase.units import Hartree as Ha

from gpaw import GPAW, FermiDirac
from gpaw import PW
from gpaw.response.g0w0 import G0W0


a = 3.567
atoms = bulk('C', 'diamond', a=a)

k = 2

calc = GPAW(mode=PW(600),
            parallel={'domain': 1},
            kpts={'size': (k, k, k), 'gamma': True},
            xc='LDA',
            occupations=FermiDirac(0.001),
            txt='C_converged_mpa.txt')

atoms.calc = calc
atoms.get_potential_energy()

calc.diagonalize_full_hamiltonian()
calc.write('C_converged_mpa.gpw', 'all')

for npols in [1, 8]:
    gw = G0W0(calc='C_converged_mpa.gpw',
              kpts=[0],
              bands=(3, 5),
              ecut=400,
              ecut_extrapolation=True,
              integrate_gamma='WS',
              mpa={'npoles': npols,
                   'varpi': Ha,
                   'eta0': 1e-10,
                   'eta_rest': 0.1 * Ha,
                   'wrange': [0, 0 if npols == 1 else 200],
                   'alpha': 1},
              filename=f'C-g0w0_mp{npols}')

    results = gw.calculate()
    direct_gap = results['qp'][0, 0, 1] - results['qp'][0, 0, 0]
    print(f'Direct gap mp{npols}:', direct_gap)
