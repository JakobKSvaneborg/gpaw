import numpy as np
from gpaw.response.chi0 import Chi0Calculator, get_frequency_descriptor
import pytest
from gpaw.response.pair import get_gs_and_context
from gpaw.mpi import world
from gpaw.response.bse import BSE, BSE_Plus
from gpaw.response.df import Chi0DysonEquations
from gpaw.response.coulomb_kernels import CoulombKernel


@pytest.mark.response
def test_bse_plus(in_tmp_dir, gpw_files, monkeypatch):
    """
    This test makes a BSE plus calculation with the BSE_Plus class and
    manually to test that the BSE_Plus code is working. It tests that the
    assertion work.
    """
    monkeypatch.chdir(in_tmp_dir)
    gs, context = get_gs_and_context(
        gpw_files['sic_pw'], txt=None, world=world, timer=None)

    ecut = 25
    eshift = 0.2
    eta = 0.1
    q_c = [0.0, 0.0, 0.0]
    bse_valence_bands = range(3, 4)
    bse_conduction_bands = range(4, 6)
    bse_nbands = 8
    rpa_nbands = 8
    bse = BSE(gpw_files['sic_pw'], ecut=ecut,
              q_c=q_c,
              valence_bands=bse_valence_bands,
              conduction_bands=bse_conduction_bands,
              eshift=eshift,
              mode='BSE',
              nbands=bse_nbands)

    w_w = np.array([-3, 0, 6])
    wd = get_frequency_descriptor(w_w, gs=gs)

    chi0calc_small = Chi0Calculator(gs, context,
                                    wd=wd,
                                    nbands=slice(3, 6),
                                    intraband=False,
                                    hilbert=False,
                                    eta=eta,
                                    ecut=ecut,
                                    eshift=eshift)

    chi0calc_large = Chi0Calculator(gs, context,
                                    wd=wd,
                                    nbands=rpa_nbands,
                                    intraband=False,
                                    hilbert=False,
                                    eta=eta,
                                    ecut=ecut,
                                    eshift=eshift)

    bse_plus = BSE_Plus(bse_gpw=gpw_files['sic_pw'],
                        bse_valence_bands=bse_valence_bands,
                        bse_conduction_bands=bse_conduction_bands,
                        bse_nbands=bse_nbands,
                        rpa_gpw=gpw_files['sic_pw'],
                        rpa_nbands=rpa_nbands,
                        w_w=w_w,
                        eshift=eshift,
                        eta=eta,
                        q_c=q_c,
                        ecut=ecut)

    bse_plus.get_chi_wGG(optical=True,
                         chi_BSE=True,
                         chi_RPA=True,
                         bsep_name='chi_BSE_Plus_3bands',
                         bse_name='chi_BSE',
                         rpa_name='chi_RPA')
    if world.rank == 0:
        chi_BSE_plus_WGG = np.load("chi_BSE_Plus_3bands.npy")
        chi_BSE_WGG = np.load("chi_BSE.npy")
        chi_RPA_WGG = np.load("chi_RPA.npy")

    coulomb_kernel = CoulombKernel.from_gs(gs, truncation=None)

    chi_irr_BSE_WGG = bse.get_chi_wGG(eta=eta,
                                      optical=True,
                                      irreducible=True,
                                      w_w=w_w)

    chi_BSE_WGG_from_bse = bse.get_chi_wGG(eta=eta,
                                           optical=True,
                                           irreducible=False,
                                           w_w=w_w)

    chi0_data_small = chi0calc_small.calculate(q_c)
    dyson_eqs_small = Chi0DysonEquations(chi0_data_small, coulomb_kernel,
                                         xc_kernel=None, cd=gs.cd)
    chi0_small_wGG = dyson_eqs_small.get_chi0_wGG(direction='x')
    chi0_small_WGG = dyson_eqs_small.wblocks.all_gather(chi0_small_wGG)

    chi0_data_large = chi0calc_large.calculate(q_c)
    dyson_eqs_large = Chi0DysonEquations(chi0_data_large, coulomb_kernel,
                                         xc_kernel=None, cd=gs.cd)
    chi0_large_wGG = dyson_eqs_large.get_chi0_wGG(direction='x')
    chi0_large_WGG = dyson_eqs_large.wblocks.all_gather(chi0_large_wGG)

    v_G = coulomb_kernel.V(chi0_data_large.qpd)

    bare_df = dyson_eqs_large.bare_dielectric_function(direction='x')
    chi_RPA_wGG_from_df = bare_df.vchibar_symm_wGG
    chi_RPA_WGG_from_df = dyson_eqs_large.wblocks.all_gather(
        chi_RPA_wGG_from_df)
    if world.rank == 0:
        sqrtV_G = v_G**0.5
        chi_RPA_WGG_from_df /= sqrtV_G * sqrtV_G[:, np.newaxis]

        assert chi_BSE_WGG_from_bse == pytest.approx(chi_BSE_WGG,
                                                     rel=5e-2, abs=5e-2)

        assert chi_RPA_WGG_from_df == pytest.approx(chi_RPA_WGG,
                                                    rel=1e-3, abs=1e-4)

    v_G[0] = 0.0

    if world.rank == 0:
        chi_irr_BSE_plus_WGG = \
            chi_irr_BSE_WGG - chi0_small_WGG + chi0_large_WGG

        eye = np.eye(len(chi_irr_BSE_plus_WGG[1]))

        chi_BSE_plus_WGG_manuel = \
            np.linalg.solve(eye - chi_irr_BSE_plus_WGG @ np.diag(v_G),
                            chi_irr_BSE_plus_WGG)

        assert chi_BSE_plus_WGG_manuel == pytest.approx(chi_BSE_plus_WGG,
                                                        rel=1e-3, abs=1e-4)

        ref_BSE_plus = [(-0.035944653032812565 - 0.01210489304296787j),
                        (-0.0015325557438748021 - 0.005681882257833088j),
                        (-0.008360476962747542 + 0.0006972132287041919j)]

        ref_BSE = [(-0.0324945448838119 - 0.017379162560129873j),
                   (-0.00020976702039095932 - 0.0029422719254655243j),
                   (-0.0070628859974412005 + 0.0029784568548538314j)]

        ref_RPA = [(-0.027204997411298056 - 0.008582830564182742j),
                   (-0.0016731748176204374 - 0.00548611231117654j),
                   (-0.007730855455966065 - 0.0012352187727613646j)]

        for i in range(3):
            assert np.allclose(chi_BSE_plus_WGG[i, i, i + 1],
                               ref_BSE_plus[i], rtol=1e-4, atol=1e-4)
            assert np.allclose(chi_BSE_WGG[i, i, i + 1],
                               ref_BSE[i], rtol=1e-4, atol=1e-4)
            assert np.allclose(chi_RPA_WGG[i, i, i + 1],
                               ref_RPA[i], rtol=1e-4, atol=1e-4)

    # assertion error if more bands in the bse calculation
    with pytest.raises(AssertionError, match=r'Large chi0 calculation*'):
        rpa_nbands = 5
        BSE_Plus(bse_gpw=gpw_files['sic_pw'],
                 bse_valence_bands=bse_valence_bands,
                 bse_conduction_bands=bse_conduction_bands,
                 bse_nbands=bse_nbands,
                 rpa_gpw=gpw_files['sic_pw'],
                 rpa_nbands=rpa_nbands,
                 w_w=w_w,
                 eshift=eshift,
                 eta=eta,
                 q_c=q_c,
                 ecut=ecut)
