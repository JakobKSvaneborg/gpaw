from pathlib import Path

import pytest
from ase import Atoms

from gpaw.benchmark.performance_index import main, work
from gpaw.benchmark.systems import systems
from gpaw.mpi import world


@pytest.mark.serial
@pytest.mark.parametrize('name', systems)
def test_systems(name):
    atoms = systems[name]()
    assert isinstance(atoms, Atoms)


def test_pw_benchmark(in_tmp_dir):
    if world.rank == 0:
        Path('params.json').write_text(
            '{"mode": "pw"}')
    world.barrier()
    work('H2-0')
    main([])
    main(['.', '.'])
