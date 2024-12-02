import pytest
import numpy as np
from ase.build import bulk
from gpaw import GPAW, FermiDirac
from gpaw.response.bse import BSE
from gpaw.utilities.elpa import LibElpa
from gpaw.utilities.scalapack import have_mkl
from gpaw.mpi import world


@pytest.mark.response
@pytest.mark.skipif(world.size == 1
                    or (not have_mkl() and not LibElpa.have_elpa()),
                    reason='requires mkl and elpa and parallel run')
def test_response_bse_diagonalization(in_tmp_dir, scalapack):
    GS = 1
    bse = 1

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
        if LibElpa.have_elpa():
            with pytest.warns():
                w2_T, v2_Rt, _ = bse_matrix.diagonalize_tammdancoff(
                    bse=bse, backend='elpa')
                assert w_T == pytest.approx(w2_T, abs=1e-3)

        if have_mkl():
            w3_T, v3_Rt, _ = bse_matrix.diagonalize_tammdancoff(bse=bse,
                                                                backend='mkl')

            argsort = np.argsort(w3_T)
            assert w_T == pytest.approx(w3_T[argsort], abs=1e-3)
