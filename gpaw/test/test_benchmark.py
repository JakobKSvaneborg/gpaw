from pathlib import Path

import pytest
from ase import Atoms
from gpaw.benchmark.pw_mode_check import summary, work
from gpaw.benchmark.systems import systems


@pytest.mark.serial
@pytest.mark.parametrize('name', systems)
def test_systems(name):
    atoms = systems[name]()
    assert isinstance(atoms, Atoms)


def test_pw_benchmark(in_tmp_dir):
    Path('params.json').write_text(
        '{"mode": "pw"}')
    work('H2')
    summary([Path(), Path()], mode=3)
