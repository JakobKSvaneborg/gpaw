import pytest

from gpaw.response import ResponseGroundStateAdapter
from gpaw.test.gpwfile import response_band_cutoff


@pytest.mark.response
@pytest.mark.parametrize('identifier', list(response_band_cutoff))
def test_qsymmetries(gpw_files, identifier):
    gs = ResponseGroundStateAdapter.from_gpw_file(gpw_files[identifier])
    kpts = gs._calc.parameters.get('kpts', {})
    if 'gamma' in kpts and kpts['gamma']:
        # If the ground state is Γ-centered, all IBZ k-points are valid
        # q-points as well (autocommensurate) and we check that the q-point
        # symmetry analyzer reproduces the symmetries of the ground state.
        pass  # XXX
    else:
        # If the ground state isn't Γ-centered, we simply check that a "noisy"
        # Γ-point q vector recovers all symmetries of the system
        pass  # XXX
