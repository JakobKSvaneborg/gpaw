import pytest
import numpy as np
from ase.build import bulk
from gpaw import GPAW, FermiDirac
from gpaw.response.bse import BSE


@pytest.mark.response
def test_response_bse_diagonalization(in_tmp_dir, scalapack):
    GS = 1
    bse = 1
    check = 1

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
                  nbands=8)

        bse_matrix = bse.calculate(optical=True)
        w_T, v_Rt, exclude_S = \
            bse_matrix.diagonalize_tammdancoff(bse=bse, backend='scalapack')
        w2_T, v2_Rt, _ = bse_matrix.diagonalize_tammdancoff(bse=bse,
                                                            backend='elpa')
        w3_T, v3_Rt, _ = bse_matrix.diagonalize_tammdancoff(bse=bse,
                                                            backend='mkl')

    if check:
        for w in w_T:
            assert np.isclose(w, w2_T, atol=1e-6).any()
            assert np.isclose(w, w3_T, atol=1e-6).any()
