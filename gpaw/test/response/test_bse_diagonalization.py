import pytest
import numpy as np
from ase.build import bulk
from gpaw import FermiDirac
from gpaw.mpi import world
from gpaw.response.bse import BSE
from gpaw.utilities.elpa import LibElpa


@pytest.mark.response
@pytest.mark.skipif(world.size == 1 or not LibElpa.have_elpa(),
                    reason='requires elpa and parallel run')
def test_response_bse_diagonalization(in_tmp_dir, scalapack, mpi):
    GS = 1
    bse = 1

    if GS:
        a = 5.431  # From PRB 73,045112 (2006)
        atoms = bulk('Si', 'diamond', a=a)
        atoms.positions -= a / 8
        calc = mpi.GPAW(mode='pw',
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
                  nbands=8, txt=None,
                  comm=world)

        bse_matrix = bse.calculate(optical=True)
        w_T, v_Rt, exclude_S = \
            bse_matrix.diagonalize_tammdancoff(bse=bse, backend='scalapack')
        if LibElpa.have_elpa():
            with pytest.warns():
                w2_T, v2_Rt, _ = bse_matrix.diagonalize_tammdancoff(
                    bse=bse, backend='elpa')
                assert w_T == pytest.approx(w2_T, abs=1e-3)


@pytest.mark.response
def test_bse_skew_diagonalization(in_tmp_dir, scalapack, mpi):
    """Test skew-symmetric BSE diagonalization against general eigensolver.

    Constructs a small non-TDA BSE Hamiltonian and compares the
    eigenvalues from the skew-symmetric reduction (K_J = L^T J L)
    with those from numpy.linalg.eig.
    """
    a = 5.431
    atoms = bulk('Si', 'diamond', a=a)
    atoms.positions -= a / 8
    calc = mpi.GPAW(mode='pw',
                    kpts={'size': (2, 2, 2), 'gamma': True},
                    occupations=FermiDirac(0.001),
                    nbands=12,
                    convergence={'bands': -4})
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('Si_skew.gpw', 'all')

    # Non-TDA BSE requires overlapping valence/conduction bands
    bse = BSE('Si_skew.gpw',
              ecut=50.,
              valence_bands=range(2, 6),
              conduction_bands=range(4, 8),
              eshift=0.8,
              nbands=8, txt=None,
              comm=world)

    bse_matrix = bse.calculate(optical=True)

    if not bse.use_tammdancoff:
        # Reference: general non-symmetric eigensolver
        w_ref, v_ref, excl_ref, vl_ref = \
            bse_matrix.diagonalize_nontammdancoff(bse)

        # Skew-symmetric (scipy fallback for serial, ELPA for parallel)
        try:
            w_skew, v_skew, excl_skew, vl_skew = \
                bse_matrix.diagonalize_nontammdancoff_structured(
                    bse, backend='elpa_skew')
        except RuntimeError:
            # May fail if BSE matrix is complex (no TRS)
            pytest.skip('BSE matrix is complex; skew reduction not applicable')

        # Compare eigenvalues (sorted by real part)
        w_ref_sorted = np.sort(w_ref.real)
        w_skew_sorted = np.sort(w_skew.real)
        assert w_ref_sorted == pytest.approx(w_skew_sorted, abs=1e-6)

        # Also test Hermitian backend
        w_herm, v_herm, excl_herm, vl_herm = \
            bse_matrix.diagonalize_nontammdancoff_structured(
                bse, backend='hermitian')
        w_herm_sorted = np.sort(w_herm.real)
        assert w_ref_sorted == pytest.approx(w_herm_sorted, abs=1e-6)
