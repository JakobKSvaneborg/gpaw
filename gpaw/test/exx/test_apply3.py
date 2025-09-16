from gpaw.new.pw.hybridsk import PWHybridHamiltonianK, Psit
from gpaw.core import UGDesc, PWDesc
from gpaw.setup import Setups
from gpaw.core.atom_arrays import AtomDistribution, AtomArraysLayout
import numpy as np
from gpaw.mpi import world


def test_apply3():
    a = 5.0
    c = 4.0
    grid = UGDesc.from_cell_and_grid_spacing([a, a, c, 90, 90, 120], 0.3)
    pw = PWDesc(cell=grid.cell, ecut=grid.ekin_max() * 0.8)
    N = 10
    relpos_ac = np.empty((N, 3))
    relpos_ac[:] = np.linspace(0, 1.0, N, False)[:, np.newaxis]
    xc = type('XC', (), {'exx_fraction': 0.25,
                         'exx_omega': 0.2})()
    ham = PWHybridHamiltonianK(
        grid, pw, xc,
        Setups([6] * N, 'paw', {}, 'LDA'),
        relpos_ac,
        atomdist=AtomDistribution.from_number_of_atoms(N),
        log=print,
        kpt_comm=world,
        band_comm=world,
        comm=world)
    pwk = pw.new(kpt=[0.5, 0.5, 0.0])
    n1 = 15
    n2 = 153
    psit1 = Psit(
        ut_nR=grid.empty(n1).data,
        P_ani=AtomArraysLayout().empty(n1),
        f_n=np.ones(n1),
        kpt_c=np.ndarray([0.6, 0.6, 0.6]),
        Q_aniL={a: np.ones((n1, 13, 9), complex) for a inb range(N)},
        spin=0)
    v_G = pwg.empty()
    v_G.data[:] = 1.0 / pwg.ekin_G # real?  Use np.ndarray
    ut2_nR = grid.empty(n2)
    ham._apply3(
        v_G.
        psit1,
        ut2_nR,
        P2_ani,
        Htpsit2_nG,
        V2_ani,
        f2_n=np.ones(n2),
        calculate_energy=True)
    