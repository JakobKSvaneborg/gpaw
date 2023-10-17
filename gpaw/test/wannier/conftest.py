import shutil
import pytest


@pytest.fixture
def wannier90(scope='session'):
    wannier90 = shutil.which('wannier90.x')
    if wannier90 is None:
        pytest.skip('no wannier90.x executable')

    # Actually we should /return/ the path, and we should have stored
    # it in a kind of config rather than hardcoding the executable.
    #
    # TODO: return wannier90 path and pass it to Wannier90 class
