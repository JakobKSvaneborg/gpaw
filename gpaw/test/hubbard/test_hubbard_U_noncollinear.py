import pytest

import numpy as np
from ase.units import Ha

from gpaw.hubbard import hubbard


def test_hubbard_U_noncollinear():
    rng = np.random.default_rng(seed=555)

    # Generate random data simulating a setup with
    # s and p bounded and unbounded partial waves.

    D_sii = (rng.random([4, 8, 8], dtype=np.float64)
             + 1j * rng.random([4, 8, 8], dtype=np.float64))
    # Make Hermitian
    D_sii = (D_sii + np.transpose(D_sii.conj(), (0, 2, 1))) / 2
    n_j = [2, 2, -1, -1]
    l_j = [0, 1, 0, 1]

    # Hubbard correction on the p orbitals
    l = 1
    U = 4  # eV

    # Generate inner products for the p-orbitals
    N0_q = np.zeros(len(l_j) * (len(l_j) + 1) // 2)
    N0_q[4] = 0.9  # Bounded-bounded
    N0_q[6] = 1.1  # Bounded-unbounded
    N0_q[9] = 1.2  # Unbounded-unbounded

    eU, V_sii = hubbard(D_sii, U / Ha, l, l_j, n_j, N0_q, scale=True)
    eU_onlyreal, _ = hubbard(D_sii.real, U / Ha, l, l_j, n_j, N0_q, scale=True)
    assert eU.imag == pytest.approx(0, abs=1e-10)

    # Assert that disregarding the complex values will alter the energy
    assert eU == pytest.approx(-7.825104, abs=1.e-2)
    assert eU_onlyreal == pytest.approx(-7.546553, abs=1.e-2)

    # Now, we check that the Hubbard Hamiltonian is calculated correctly by
    # comparing it with an energy derivative w.r.t. a density matrix element
    # calculated through finite difference.

    diff = 1e-5

    D1_sii = D_sii.copy()
    D1_sii[0, 1, 2] += diff / 2

    eU1, _ = hubbard(D1_sii, U / Ha, l, l_j, n_j, N0_q, scale=True)

    D2_sii = D_sii.copy()
    D2_sii[0, 1, 2] -= diff / 2

    eU2, _ = hubbard(D2_sii, U / Ha, l, l_j, n_j, N0_q, scale=True)

    # print(eU1)
    # print(eU2)
    # print(V_sii[0, 1:4, 1:4])
    assert (eU1 - eU2) / diff == pytest.approx(V_sii[0, 1, 2], abs=1e-8)
