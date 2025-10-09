from gpaw.new.pw.hybrids import PWHybridHamiltonian, Psit
from gpaw.core import UGDesc, PWDesc
from gpaw.setup import Setups
from gpaw.core.atom_arrays import AtomDistribution, AtomArraysLayout
import numpy as np
from gpaw.mpi import world
from time import time


def test_apply3(a=5.0, N=10, dtype=float):
    c = 4.0
    grid = UGDesc.from_cell_and_grid_spacing([a, a, c, 90, 90, 120], 0.3,
                                             dtype=dtype)
    pw = PWDesc(cell=grid.cell, ecut=grid.ekin_max() * 0.8, dtype=dtype)
    relpos_ac = np.empty((N, 3))
    relpos_ac[:] = np.linspace(0, 1.0, N, False)[:, np.newaxis]
    xc = type('XC', (), {'exx_fraction': 0.25,
                         'exx_omega': 0.2})()
    ham = PWHybridHamiltonian(
        grid, pw, xc,
        Setups([6] * N, 'paw', {}, 'LDA'),
        relpos_ac,
        atomdist=AtomDistribution.from_number_of_atoms(N),
        log=print,
        kpt_comm=world,
        band_comm=world,
        comm=world)
    if dtype == float:
        pw2 = pw
        pw12 = pw
        kpt1_c = np.zeros(3)
    else:
        kpt1_c = np.array([0.5, 0.5, 0.0])
        kpt2_c = np.array([0.25, 0.25, 0.0])
        pw2 = pw.new(kpt=[0.25, 0.25, 0.0], dtype=dtype)
        pw12 = pw.new(kpt=kpt2_c - kpt1_c, dtype=dtype)
    n1 = 35
    n2 = 153
    layout = AtomArraysLayout([13] * N, dtype=dtype)
    psit1 = Psit(
        ut_nR=grid.empty(n1),
        P_ani=layout.empty(n1),
        f_n=np.ones(n1),
        kpt_c=kpt1_c,
        Q_aniL={a: np.ones((n1, 13, 9), dtype=dtype) for a in range(N)},
        spin=0)
    psit1.ut_nR.data[:] = 0.1
    psit1.P_ani.data[:] = 0.1
    v_G = 1.0 / (0.1 + pw12.ekin_G)
    ut2_nR = grid.empty(n2)
    P2_ani = layout.empty(n2)
    V2_ani = P2_ani.new()
    ut2_nR.data[:] = 0.1
    P2_ani.data[:] = 1.0
    V2_ani.data[:] = 0.0
    Htpsit2_nG = pw2.zeros(n2)
    print(psit1.ut_nR.data.shape)
    print(Htpsit2_nG.data.shape)
    ham.nbzk = 1
    t = time()
    e = ham._apply3(
        pw12,
        v_G,
        psit1,
        ut2_nR,
        P2_ani,
        Htpsit2_nG,
        V2_ani,
        f2_n=np.ones(n2),
        calculate_energy=True,
        F1_av=None)
    t = time() - t
    print(e)
    print(t)


def mmmi_test(A=16):
    from gpaw.utilities.blas import mmm
    from time import time
    r_nG = np.ones((A * 3, 2300), dtype=complex) + 0.7j
    g_GI = np.ones((2300, A * 9), dtype=complex) + 0.3j
    Q_nI = np.ones((A * 3, A * 9), dtype=complex) - 0.1j
    t = time()
    # mmm(1.0, r_nG, 'N', g_GI, 'N', 0.0, Q_nI)
    mmm(0.23, Q_nI, 'N', g_GI, 'T', 1.0, r_nG)
    return (time() - t) / A**2


if __name__ == '__main__':
    test_apply3(4.5, 8, dtype=complex)
    if 0:
        T = 0.0
        for i in range(50):
            T += mmmi_test(8)
        print(T / 50)
