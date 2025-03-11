import numpy as np
import pytest

from gpaw.response import ResponseContext, ResponseGroundStateAdapter
from gpaw.response.symmetry import QSymmetryAnalyzer
from gpaw.test.gpwfile import response_band_cutoff


@pytest.mark.response
@pytest.mark.parametrize('identifier', list(response_band_cutoff))
def test_qsymmetries(gpw_files, identifier):
    # Set up basic response code objects
    gs = ResponseGroundStateAdapter.from_gpw_file(gpw_files[identifier])
    context = ResponseContext()
    qsymmetry = QSymmetryAnalyzer()

    # Test symmetry analysis
    kpts = gs._calc.parameters.get('kpts', {})
    if 'gamma' in kpts and kpts['gamma']:
        # If the ground state is Γ-centered, all IBZ k-points are valid
        # q-points as well (autocommensurate) and we check that the q-point
        # symmetry analyzer reproduces the symmetries of the ground state.
        pass  # XXX
    else:
        # If the ground state isn't Γ-centered, we simply check that a "noisy"
        # Γ-point q vector recovers all symmetries of the system
        # All symmetries:
        symmetry = gs.kd.symmetry
        ndirect = len(symmetry.op_scc)
        nindirect = ndirect * (1 - symmetry.has_inversion)
        # "Noisy" q-point:
        q_c = (np.random.rand(3) - 0.5) * 1e-15
        qsymmetries, _ = qsymmetry.analyze(q_c, gs.kpoints, context)
        assert qsymmetries.ndirect == ndirect, \
            f'{q_c}: {qsymmetries.ndirect} / {ndirect}'
        assert qsymmetries.nindirect == nindirect, \
            f'{q_c}: {qsymmetries.nindirect} / {nindirect}'
