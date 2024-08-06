import numpy as np
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.typing import Array2D


def get_wannier_integrals(ibzwfs: IBZWaveFunctions,
                          grid,
                          s: int,
                          k: int,
                          k1: int,
                          G_c,
                          nbands=None) -> Array2D:
    """Calculate integrals for maximally localized Wannier functions."""
    assert s <= ibzwfs.nspins
    # XXX not for the kpoint/spin parallel case
    assert ibzwfs.comm.size == 1
    wfs = ibzwfs.wfs_qs[k][s].to_uniform_grid_wave_functions(grid, None)
    wfs1 = ibzwfs.wfs_qs[k1][s].to_uniform_grid_wave_functions(grid, None)
    # Get pseudo part
    psit_nR = wfs.psit_nX.data
    psit1_nR = wfs1.psit_nX.data
    Z_nn = grid._gd.wannier_matrix(psit_nR, psit1_nR, G_c, nbands)
    # Add corrections
    add_wannier_correction(Z_nn, G_c, wfs, wfs1, nbands)
    grid.comm.sum(Z_nn)
    return Z_nn


def add_wannier_correction(Z_nn, G_c, wfs, wfs1, nbands):
    r"""Calculate the correction to the wannier integrals.

    See: (Eq. 27 ref1)::

                      -i G.r
        Z   = <psi | e      |psi >
         nm       n             m

                       __                __
               ~      \              a  \     a*   a    a
        Z    = Z    +  ) exp[-i G . R ]  )   P   dO    P
         nmx    nmx   /__            x  /__   ni   ii'  mi'

                       a                 ii'

    Note that this correction is an approximation that assumes the
    exponential varies slowly over the extent of the augmentation sphere.

    ref1: Thygesen et al, Phys. Rev. B 72, 125119 (2005)
    """
    P_ani = wfs.P_ani
    P1_ani = wfs1.P_ani
    for a, P_ni in P_ani.items():
        P_ni = P_ani[a][:nbands]
        P1_ni = P1_ani[a][:nbands]
        dO_ii = wfs.setups[a].dO_ii
        e = np.exp(-2.j * np.pi * np.dot(G_c, wfs.fracpos_ac[a]))
        Z_nn += e * P_ni.conj() @ dO_ii @ P1_ni.T
