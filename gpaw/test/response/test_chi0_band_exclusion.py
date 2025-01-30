from gpaw.mpi import world
from gpaw.response.pair import get_gs_and_context
from gpaw.response.chi0 import Chi0Calculator, get_omegamax
import pytest
from gpaw.response.frequencies import FrequencyDescriptor


@pytest.mark.response
def test_chi0_band_exclusion(in_tmp_dir, gpw_files):
    """Testing the removal of the lowest three valence bands in a chi0
    calculation for Ni. This is done by comparing two chi0 calculation: one
    includs all bands but limits the frequency grid to exclude the transitions
    from the 3 lowest valence bands, and the other explicitly excludes
    these bands but extends the grid to cover their transition range.
    The real part is obtained via a Hilbert transform"""

    gs, context = get_gs_and_context(
        gpw_files['ni_pw'], txt=None, world=world, timer=None)

    ecut = 40
    eta = 0.1
    nbands_max = 14

    omegamax2 = get_omegamax(gs, nbands=slice(0, nbands_max))
    wd2 = FrequencyDescriptor.from_array_or_dict({'type': 'nonlinear',
                                                  'domega0': omegamax2 / 4000,
                                                  'omega2': 10,
                                                  'omegamax': omegamax2})
    omegamax1 = get_omegamax(gs, nbands=slice(3, nbands_max))
    wd1 = FrequencyDescriptor.from_array_or_dict({'type': 'nonlinear',
                                                  'domega0': omegamax2 / 4000,
                                                  'omega2': 10,
                                                  'omegamax': omegamax1})

    chi0calc1 = Chi0Calculator(gs, context,
                               wd=wd1, nbands=nbands_max,
                               intraband=False,
                               hilbert=True,
                               eta=eta,
                               ecut=ecut,
                               eshift=None)

    chi0_data1 = chi0calc1.calculate(q_c=[0, 0, 0])

    chi0calc2 = Chi0Calculator(gs, context,
                               wd=wd2, nbands=slice(3, nbands_max),
                               intraband=False,
                               hilbert=True,
                               eta=eta,
                               ecut=ecut,
                               eshift=None)

    chi0_data2 = chi0calc2.calculate(q_c=[0, 0, 0])

    chi0_data1_body = \
        chi0_data1.body.data_WgG
    chi0_data2_body = \
        chi0_data2.body.data_WgG

    n_freq_points = len(chi0_data1.chi0_Wvv)

    assert chi0_data1_body[:n_freq_points] == pytest.approx(
        chi0_data2_body[:n_freq_points], rel=1e-3, abs=1e-4)
    assert chi0_data1.chi0_WxvG[:n_freq_points] == pytest.approx(
        chi0_data2.chi0_WxvG[:n_freq_points], rel=1e-3, abs=1e-4)
    assert chi0_data1.chi0_Wvv[:n_freq_points] == pytest.approx(
        chi0_data2.chi0_Wvv[:n_freq_points], rel=1e-3, abs=1e-4)

    # test assertion error when n1 >= n2
    n2 = gs.nocc2
    m1 = gs.nocc1
    with pytest.raises(AssertionError):
        chi0calc = Chi0Calculator(gs, context,
                                  wd=wd2, nbands=slice(n2, nbands_max),
                                  intraband=False,
                                  hilbert=True,
                                  eta=eta,
                                  ecut=ecut,
                                  eshift=None)
        chi0calc.calculate(q_c=[0, 0, 0])

    with pytest.raises(AssertionError):
        chi0calc = Chi0Calculator(gs, context,
                                  wd=wd2, nbands=slice(n2 + 1, nbands_max),
                                  intraband=False,
                                  hilbert=True,
                                  eta=eta,
                                  ecut=ecut,
                                  eshift=None)
        chi0calc.calculate(q_c=[0, 0, 0])

    # test assertion error when n1 > m1
    with pytest.raises(AssertionError):
        chi0calc = Chi0Calculator(gs, context,
                                  wd=wd2, nbands=slice(m1 + 1, nbands_max),
                                  intraband=False,
                                  hilbert=True,
                                  eta=eta,
                                  ecut=ecut,
                                  eshift=None)
        chi0calc.calculate(q_c=[0, 0, 0])

    # test assertion error if step size is not none or 1
    with pytest.raises(AssertionError):
        chi0calc = Chi0Calculator(gs, context,
                                  wd=wd2, nbands=slice(3, nbands_max, 3),
                                  intraband=False,
                                  hilbert=True,
                                  eta=eta,
                                  ecut=ecut,
                                  eshift=None)
        chi0calc.calculate(q_c=[0, 0, 0])

    # test assertion error if n1 is negative
    with pytest.raises(AssertionError):
        chi0calc = Chi0Calculator(gs, context,
                                  wd=wd2, nbands=slice(-1, nbands_max),
                                  intraband=False,
                                  hilbert=True,
                                  eta=eta,
                                  ecut=ecut,
                                  eshift=None)
        chi0calc.calculate(q_c=[0, 0, 0])

    # test assertion error if m2 is negative
    with pytest.raises(AssertionError):
        chi0calc = Chi0Calculator(gs, context,
                                  wd=wd2, nbands=slice(3, -2),
                                  intraband=False,
                                  hilbert=True,
                                  eta=eta,
                                  ecut=ecut,
                                  eshift=None)
        chi0calc.calculate(q_c=[0, 0, 0])
