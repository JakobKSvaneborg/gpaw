import numpy as np
import pytest

from gpaw import GPAW


@pytest.mark.do
def test_orthonormalizations_lcao(in_tmp_dir, gpw_files):
    """
    Test Loewdin and Gram-Schmidt orthonormalization
    of orbitals in LCAO
    :param in_tmp_dir:
    :return:
    """
    calc = GPAW(gpw_files['h3_orthonorm_lcao'])
    atoms = calc.atoms
    atoms.calc = calc

    for type in ['loewdin', 'gramschmidt']:
        atoms.positions[0] += 0.1
        calc.initialize_positions(atoms)
        for kpt in calc.wfs.kpt_u:
            calc.wfs.orthonormalize(kpt, type=type)
            overlaps = np.dot(kpt.C_nM.conj(),
                              np.dot(kpt.S_MM, kpt.C_nM.T))
            assert overlaps == pytest.approx(np.identity(3), abs=1e-10)
