import pytest
import numpy as np
from ase.build import bulk
from gpaw import GPAW, FermiDirac
from gpaw.response.bse import BSE
import matplotlib.pyplot as plt

@pytest.mark.response
def test_response_bse_diagonalization(in_tmp_dir, scalapack):
    GS = 1
    bse = 1
    check = 1
    plot = 1

    if GS:
        a = 5.431  # From PRB 73,045112 (2006)
        atoms = bulk('Si', 'diamond', a=a)
        atoms.positions -= a / 8
        calc = GPAW(mode='pw',
                    kpts={'size': (2, 2, 2), 'gamma': True},
                    occupations=FermiDirac(0.001),
                    nbands=12,
                    convergence={'bands': -4})
        atoms.calc = calc
        atoms.get_potential_energy()
        calc.write('Si.gpw', 'all')

    if bse:
        eshift = 0.8
        bse = BSE('Si.gpw',
                  ecut=50.,
                  valence_bands=range(4),
                  conduction_bands=range(4, 8),
                  eshift=eshift,
                  nbands=8, txt=None)

        bse_matrix = bse.calculate(optical=True)
        w_T, v_Rt, exclude_S = \
            bse_matrix.diagonalize_tammdancoff(bse=bse, backend='scalapack')
        w2_T, v2_Rt, _ = bse_matrix.diagonalize_tammdancoff(bse=bse,
                                                            backend='elpa')
        w3_T, v3_Rt, _ = bse_matrix.diagonalize_tammdancoff(bse=bse,
                                                         backend='mkl')
        if plot:
            print('w_T.shape', w_T.shape)
            print('w2_T.shape', w2_T.shape)
            print('w3_T.shape', w3_T.shape)
            x = np.arange(len(w_T))
            argsort = np.argsort(w3_T.real)
            plt.scatter(x, w_T.real, label='scala', marker='x')
            plt.scatter(x, w2_T.real, label='elpa', marker='*')
            plt.scatter(x, w3_T.real[argsort], label='mkl', marker='.')
            plt.legend()
            plt.savefig('Ws.png')
            plt.close()
            eig_data = (w_T, v_Rt, exclude_S)
            _, C1_T = bse.get_spectral_weights(eig_data, bse.df_S, mode_c=None)

            eig_data = (w2_T, v2_Rt, exclude_S)
            _, C2_T = bse.get_spectral_weights(eig_data, bse.df_S, mode_c=None)

            eig_data = (w3_T, v3_Rt, exclude_S)
            _, C3_T = bse.get_spectral_weights(eig_data, bse.df_S, mode_c=None)

            print('C1_T.shape', C1_T.shape)
            print('C2_T.shape', C2_T.shape)
            print('C3_T.shape', C3_T.shape)
            plt.figure()
            plt.scatter(x, C1_T.real, label='scala', marker='x')
            plt.scatter(x, C2_T.real, label='elpa', marker='*')
            plt.scatter(x, C3_T.real[argsort], label='mkl', marker='.')
            plt.legend()
            plt.savefig('Cs.png')

            plt.figure()
            plt.scatter(x, C1_T.imag, label='scala', marker='x')
            plt.scatter(x, C2_T.imag, label='elpa', marker='*')
            plt.scatter(x, C3_T.imag[argsort], label='mkl', marker='.')
            plt.legend()
            plt.savefig('Cs-imag.png')

            plt.figure()
            plt.scatter(w_T, np.abs(C1_T), label='scala', marker='x')
            plt.scatter(w2_T, np.abs(C2_T), label='elpa', marker='*')
            plt.scatter(w3_T[argsort], np.abs(C3_T)[argsort], label='mkl', marker='.')
            plt.legend()
            plt.savefig('W-Cs.png')

            plt.figure()
            plt.semilogy(x, np.abs(C1_T - C2_T), label='elpa err', marker='x')
            plt.semilogy(x, np.abs(C1_T - C3_T[argsort]), label='mkl err', marker='.')
            plt.legend()
            plt.savefig('Cs-err.png')

            plt.figure()
            plt.semilogy(w_T, np.abs(C1_T - C2_T), label='elpa err', marker='x')
            plt.semilogy(w_T, np.abs(C1_T - C3_T[argsort]), label='mkl err', marker='.')
            plt.legend()
            plt.savefig('W-Cs-err.png')

    if check:
        assert w_T == pytest.approx(w2_T, abs=1e-3)
        assert w_T == pytest.approx(w3_T[argsort], abs=1e-3)
