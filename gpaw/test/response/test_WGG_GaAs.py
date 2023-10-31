import pytest
import numpy as np
from gpaw.response.coulomb_kernels import CoulombKernel
from gpaw.response.pair import get_gs_and_context
from gpaw import GPAW, PW, FermiDirac
from gpaw.mpi import world
from gpaw.response.chi0 import Chi0
from gpaw.response.screened_interaction import initialize_w_calculator
from ase.build import bulk
from ase import Atoms


@pytest.mark.response
def test_Wsymm(in_tmp_dir, scalapack):
    def calc_gs(symm):
        k = 2
        cell = bulk('Ga', 'fcc', a=5.68).cell
        a = Atoms('GaAs', cell=cell, pbc=True,
                  scaled_positions=((0, 0, 0), (0.25, 0.25, 0.25)))

        # First with symmetry off
        calc = GPAW(mode=PW(400),
                    xc='LDA',
                    occupations=FermiDirac(width=0.01),
                    convergence={'bands': -4},
                    symmetry=symm,
                    kpts={'size': (k, k, k), 'gamma': True},
                    txt='gs_GaAs.txt')

        a.calc = calc
        a.get_potential_energy()
        if symm == 'off':
            calc.write('GaAs.gpw', mode='all')
        else:
            calc.write('GaAs_symm.gpw', mode='all')
        return calc

    def get_IBZ_k(calc):
        gs = calc.gs_adapter()
        qclist = gs.kd.ibzk_kc
        return qclist

    def calc_W(seed, q_c_list):
        omega = np.array([0])
        chi0calc = Chi0(seed + '.gpw',
                        frequencies=omega,
                        hilbert=False,
                        ecut=100,
                        txt='test.log',
                        intraband=False)
        txt = 'out.txt'
        gs, wcontext = get_gs_and_context(
            seed + '.gpw',
            txt, world=world,
            timer=None)
        truncation = None
        coulomb = CoulombKernel.from_gs(gs, truncation=truncation)
        wcalc = initialize_w_calculator(chi0calc,
                                        wcontext,
                                        coulomb=coulomb,
                                        integrate_gamma=0)
        Wlist = []
        qlist = []
        for iq, q_c in enumerate(q_c_list):
            chi0 = chi0calc.calculate(q_c)
            W_wGG = wcalc.calculate_W_wGG(chi0,
                                          fxc_mode='GW',
                                          only_correlation=False)
            Wlist.append(W_wGG)
            qlist.append(q_c)
        return Wlist, qlist

    for symm in ['off', {}]:
        # calc gs with and without symmetry
        calc = calc_gs(symm)
        # get list of Q in IBZ with symmetry
        q_c_list = get_IBZ_k(calc)

    # calc W_GG
    WGG_nosymm, qnosymm = calc_W('GaAs', q_c_list)
    WGG_symm, qsymm = calc_W('GaAs_symm', q_c_list)

    # compare W_GG for all k in IBZ
    for iq, q_c in enumerate(q_c_list):
        assert np.allclose(qnosymm[iq], qsymm[iq])
        assert np.allclose(WGG_symm[iq], WGG_nosymm[iq])
