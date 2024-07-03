import os
import pytest
from gpaw.new.ase_interface import GPAW
from gpaw.unfold import Unfold, find_K_from_k


@pytest.mark.soc
def test_unfold_Ni(gpw_files):

    # Collinear calculation
    gpw = 'fcc_Ni_col'
    calc_col = GPAW(gpw_files[gpw],
                    parallel={'domain': 1, 'band': 1})

    pc = calc_col.atoms.get_cell(complete=True)
    bp = pc.get_bravais_lattice().bandpath('GX', npoints=3)

    M = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    Kpts = []
    for k in bp.kpts:
        K = find_K_from_k(k, M)[0]
        Kpts.append(K)

    # Spin 0
    unfold = Unfold(name='Ni_defect_s0',
                    calc=gpw_files[gpw],
                    M=M,
                    spin=0,
                    spinorbit=False)
    e_mk, P_mk = unfold.get_spectral_weights(bp.kpts)
    N0 = len(e_mk)
    assert P_mk == pytest.approx(1, abs=1.0e-6)

    # Spin 1
    unfold = Unfold(name='Ni_defect_s1',
                    calc=gpw_files[gpw],
                    M=M,
                    spin=1,
                    spinorbit=False)
    e_mk, P_mk = unfold.get_spectral_weights(bp.kpts)
    N1 = len(e_mk)
    assert P_mk == pytest.approx(1, abs=1.0e-6)

    # Full bands including nscf spin-orbit
    unfold = Unfold(name='Ni_defect_soc',
                    calc=gpw_files[gpw],
                    M=M,
                    spinorbit=True)
    e_mk, P_mk = unfold.get_spectral_weights(bp.kpts)
    Nm = len(e_mk)
    assert P_mk == pytest.approx(1, abs=1.0e-6)
    assert Nm == N0 + N1

    # Non-collinear calculation with self-consistent spin–orbit
    gpw = 'fcc_Ni_ncolsoc'
    calc_ncol = GPAW(gpw_files[gpw],
                     parallel={'domain': 1, 'band': 1})
    pc = calc_ncol.atoms.get_cell(complete=True)

    bp = pc.get_bravais_lattice().bandpath('GX', npoints=3)

    M = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    Kpts = []
    for k in bp.kpts:
        K = find_K_from_k(k, M)[0]
        Kpts.append(K)

    unfold = Unfold(name='Ni_defect_nc',
                    calc=gpw_files[gpw],
                    M=M)
    e_mk, P_mk = unfold.get_spectral_weights(bp.kpts)
    assert P_mk == pytest.approx(1, abs=1.0e-6)

    os.system('rm weights_Ni_defect_*pckl')
