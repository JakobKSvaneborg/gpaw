import os
from pathlib import Path

from ase.io import read


def test_fileio_read_old_gpw(mpi):
    # XXX This test never runs because we don't have this variable anymore.
    dir = os.environ.get('GPW_TEST_FILES')
    if dir:
        for f in (Path(dir) / 'old').glob('*.gpw'):
            print(f)
            calc = mpi.GPAW(str(f))
            calc.get_fermi_level()
            read(f)
