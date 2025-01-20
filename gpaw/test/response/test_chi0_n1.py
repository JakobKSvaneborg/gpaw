import numpy as np
import pytest
from gpaw.mpi import world
from gpaw.response.coulomb_kernels import CoulombKernel
from gpaw.response.pair import get_gs_and_context
from gpaw.response.df import Chi0DysonEquations
from gpaw.response.chi0 import Chi0Calculator, get_frequency_descriptor


@pytest.mark.response
def test_chi0_n1(in_tmp_dir, gpw_files):

    gs, context = get_gs_and_context(
        gpw_files['bn_pw'], txt=None, world=world, timer=None)

    nbands = 10

    wd = get_frequency_descriptor(np.array([-3, 0, 6]), gs=gs, nbands=nbands)

    chi0calc = Chi0Calculator(gs, context,
                              wd=wd, n1=3,
                              nbands=nbands,
                              intraband=False,
                              hilbert=False,
                              eta=0.2,
                              ecut=50,
                              eshift=None)

    chi0_data = chi0calc.calculate(q_c=[0, 0, 0])

    coulomb_kernel = CoulombKernel.from_gs(gs, truncation=None)

    dyson_eqs = Chi0DysonEquations(chi0_data, coulomb_kernel, None, gs.cd)

    chi0_wGG = dyson_eqs.get_chi0_wGG(direction='x')

    chi0_WGG = dyson_eqs.wblocks.all_gather(chi0_wGG)

    ref = [(-0.00133961281103788 - 0.012865342502627285j),
           (-0.0014452831931862785 - 0.00991005825917634j),
           (0.002570439550573476 + 0.007398973837800821j)]

    if world.rank == 0:
        for i, r in enumerate(ref):
            assert chi0_WGG[i, i, i + 1] == pytest.approx(r, abs=1e-02)
