from ase import __version__
from packaging.version import Version

from gpaw import __ase_version_required__


def test_ase_features_ase3k_version():
    assert Version(__version__) >= Version(__ase_version_required__)
