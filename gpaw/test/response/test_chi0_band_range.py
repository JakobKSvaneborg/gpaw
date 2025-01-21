from gpaw.mpi import world
from gpaw.response.pair import get_gs_and_context
from gpaw.response.df import Chi0DysonEquations
from gpaw.response.chi0 import Chi0Calculator
from gpaw.response.coulomb_kernels import CoulombKernel
from ase.units import Ha
import pytest
from gpaw.response.frequencies import NonLinearFrequencyDescriptor


@pytest.mark.response
def test_chi0_band_range(in_tmp_dir, gpw_files):

    gs, context = get_gs_and_context(
        gpw_files['mos2_pw'], txt=None, world=world, timer=None)
    omegamax = 25 / Ha
    wd = NonLinearFrequencyDescriptor(omegamax / 9999, 10 / Ha, omegamax)

    ecut = 15
    eta = 0.1
    nbands = 17

    chi0calc = Chi0Calculator(gs, context,
                              wd=wd, nbands=nbands,
                              intraband=False,
                              hilbert=True,
                              eta=eta,
                              ecut=ecut,
                              eshift=None)

    chi0_data = chi0calc.calculate(q_c=[0, 0, 0])

    coulomb_kernel = CoulombKernel.from_gs(gs, truncation='2D')

    dyson_eqs = Chi0DysonEquations(chi0_data, coulomb_kernel, None, gs.cd)

    chi0_wGG = dyson_eqs.get_chi0_wGG(direction='x')

    chi0_WGG_Hilbert = dyson_eqs.wblocks.all_gather(chi0_wGG)

    chi0calc = Chi0Calculator(gs, context,
                              wd=wd, band_range=slice(4, nbands),
                              intraband=False,
                              hilbert=False,
                              eta=eta,
                              ecut=ecut,
                              eshift=None)

    chi0_data = chi0calc.calculate(q_c=[0, 0, 0])

    coulomb_kernel = CoulombKernel.from_gs(gs, truncation='2D')

    dyson_eqs = Chi0DysonEquations(chi0_data, coulomb_kernel, None, gs.cd)

    chi0_wGG = dyson_eqs.get_chi0_wGG(direction='x')

    chi0_WGG_notHilbert = dyson_eqs.wblocks.all_gather(chi0_wGG)

    assert chi0_WGG_Hilbert == pytest.approx(chi0_WGG_notHilbert,
                                             rel=1e-3, abs=1e-4)
