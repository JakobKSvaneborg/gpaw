import pytest
from ase.units import Hartree

from gpaw.mpi import world
from gpaw.response.bse import BSE
from gpaw.response.pair import get_gs_and_context


@pytest.mark.response
def test_BSE_q0correction(in_tmp_dir, gpw_files):
    gs, context = get_gs_and_context(
        gpw_files['mos2_5x5_pw'], txt=None, world=world, timer=None
    )
    ecut = 25
    nbands = 12

    bse = BSE(
        gpw_files['mos2_5x5_pw'],
        ecut=ecut,
        truncation='2D',
        q0_correction=True,
        deps_max=5,
        valence_bands=2,
        conduction_bands=2,
        mode='BSE',
        nbands=nbands,
    )
    bsematrix = bse.calculate(optical=True)
    w_T, _, _ = bse.diagonalize_bse_matrix(bsematrix)
    w_T *= Hartree
    assert w_T[0] == pytest.approx(0.73127, rel=0.001)
    assert w_T[14] == pytest.approx(1.5082, rel=0.001)
