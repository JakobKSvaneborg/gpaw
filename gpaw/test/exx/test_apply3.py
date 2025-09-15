from gpaw.new.pw.hybridsk import PWHybridHamiltonianK
from gpaw.core import UGDesc, PWDesc
from gpaw.setup import Setups
from gpaw.core.atom_arrays import AtomDistribution
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
    ham._apply3(
        v_G: PWArray,
        psit1: Psit,
        ut2_nR: UGArray,
        P2_ani: AtomArrays,
        Htpsit2_nG: PWArray,
        V2_ani,
        f2_n: np.ndarray,
        calculate_energy: bool) -> float:
    