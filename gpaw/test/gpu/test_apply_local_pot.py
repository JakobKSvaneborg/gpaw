import pytest
from gpaw.core import UGDesc, PWDesc
from gpaw.new.pw.hamiltonian import apply_local_potential_gpu
from gpaw.gpu import cupy as cp


@pytest.mark.serial
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('nbands', [1, 2, 3, 5])
def test_apply_loc_pot(dtype, nbands):
    a = 1.5
    n = 4
    vt_R = UGDesc(cell=[a, a, a], size=(n, n, n)).empty(xp=cp)
    vt_R.data[:] = 2.5
    pw = PWDesc(cell=vt_R.desc.cell, ecut=4.0)
    psit_nG = pw.empty(nbands, xp=cp)
    psit_nG.data[:] = 1.2
    out_nG = pw.empty(nbands, xp=cp)
    apply_local_potential_gpu(vt_R,
                              psit_nG,
                              out_nG,
                              blocksize=3)
