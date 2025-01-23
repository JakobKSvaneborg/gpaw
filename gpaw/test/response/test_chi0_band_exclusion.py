from gpaw.mpi import world
from gpaw.response.pair import get_gs_and_context
from gpaw.response.df import Chi0DysonEquations
from gpaw.response.chi0 import Chi0Calculator
from gpaw.response.coulomb_kernels import CoulombKernel
from ase.units import Ha
import pytest
from gpaw.response.frequencies import NonLinearFrequencyDescriptor


@pytest.mark.response
def test_chi0_band_exclusion(in_tmp_dir, gpw_files):

    gs, context = get_gs_and_context(
        gpw_files['ni_pw'], txt=None, world=world, timer=None)

    ecut = 40
    eta = 0.1
    nbands = 14

    omegamax1 = 50 / Ha
    wd1 = NonLinearFrequencyDescriptor(omegamax1 / 1000, 10 / Ha, omegamax1)

    chi0calc1 = Chi0Calculator(gs, context,
                               wd=wd1, nbands=nbands,
                               intraband=False,
                               hilbert=True,
                               eta=eta,
                               ecut=ecut,
                               eshift=None)

    chi0_data1 = chi0calc1.calculate(q_c=[0, 0, 0])

    coulomb_kernel = CoulombKernel.from_gs(gs, truncation=None)

    dyson_eqs1 = Chi0DysonEquations(chi0_data1, coulomb_kernel, None, gs.cd)

    chi0_wGG1 = dyson_eqs1.get_chi0_wGG(direction='x')

    chi0_WGG = dyson_eqs1.wblocks.all_gather(chi0_wGG1)

    omegamax2 = 200 / Ha
    wd2 = NonLinearFrequencyDescriptor(omegamax2 / 4000, 10 / Ha, omegamax2)

    chi0calc2 = Chi0Calculator(gs, context,
                               wd=wd2, band_range=slice(3, nbands),
                               intraband=False,
                               hilbert=True,
                               eta=eta,
                               ecut=ecut,
                               eshift=None)

    chi0_data2 = chi0calc2.calculate(q_c=[0, 0, 0])

    dyson_eqs2 = Chi0DysonEquations(chi0_data2, coulomb_kernel, None, gs.cd)

    chi0_wGG2 = dyson_eqs2.get_chi0_wGG(direction='x')

    chi0_WGG_bands_excluded = dyson_eqs2.wblocks.all_gather(chi0_wGG2)

    assert chi0_WGG[:327, :, :] == \
        pytest.approx(chi0_WGG_bands_excluded[:327, :, :], rel=1e-3, abs=1e-4)
