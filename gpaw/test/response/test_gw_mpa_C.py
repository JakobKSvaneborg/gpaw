import numpy as np
from ase.build import bulk
from ase.units import Hartree as Ha

from gpaw import GPAW, FermiDirac
from gpaw import PW
from gpaw.response.g0w0 import G0W0


def test_diamond_mpa(in_tmp_dir):
    a = 3.567
    atoms = bulk('C', 'diamond', a=a)

    k = 2

    calc = GPAW(mode=PW(400),
                parallel={'domain': 1},
                kpts={'size': (k, k, k), 'gamma': True},
                xc='LDA',
                occupations=FermiDirac(0.001))

    atoms.calc = calc
    atoms.get_potential_energy()

    calc.diagonalize_full_hamiltonian()
    calc.write('C_converged_mpa.gpw', 'all')

    ref_results_mp1 = np.array([[[11.48389, 18.685187]]])
    ref_results_mp8 = np.array([[[11.239777, 18.591851]]])
    ref_results = {1: ref_results_mp1, 8: ref_results_mp8}

    for npols in [1, 8]:
        gw = G0W0(calc='C_converged_mpa.gpw',
                  kpts=[0],
                  bands=(3, 5),
                  ecut=200,
                  ecut_extrapolation=True,
                  integrate_gamma='WS',
                  mpa={'npoles': npols, 'wrange': [0, 200 if npols > 1 else 0],
                       'varpi': Ha, 'eta0': 1e-10, 'eta_rest': 0.1 * Ha,
                       'alpha': 1},
                  filename=f'C-g0w0_mp{npols}')

        results = gw.calculate()
        direct_gap = results['qp'][0, 0, 1] - results['qp'][0, 0, 0]
        print(f'Direct gap mp{npols}:', direct_gap)
        np.testing.assert_allclose(results['qp'], ref_results[npols],
                                   rtol=1e-02)
