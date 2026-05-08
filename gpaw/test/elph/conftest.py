import os
import pytest

from ase.build import bulk

from gpaw import GPAW
from gpaw.elph import DisplacementRunner, Supercell
from gpaw.mpi import broadcast, world

SUPERCELL = (2, 1, 1)


@pytest.fixture(scope='session')
def session_tmp_path(tmp_path_factory):
    if world.rank == 0:
        path = tmp_path_factory.mktemp('elph_session')
    else:
        path = None
    path = broadcast(path, comm=world)
    cwd = os.getcwd()
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(cwd)


def get_calc(txt, parallel={}):
    return GPAW(mode='lcao',
                basis='sz(dzp)',
                kpts={'size': (1, 2, 2), 'gamma': False},
                symmetry={'point_group': False},
                convergence={'forces': 1.e-4},
                parallel=parallel,
                txt=txt)


# elph_cache should only be calculated once per session
@pytest.fixture(scope='session')
def elph_cache(session_tmp_path):
    """Minimum elph cache for Li

    Uses 1x2x2 k-points and 2x1x1 SC to allow for parallelisaiton
    test.
    Takes 6s on 4 cores.
    """
    atoms = bulk('Li', crystalstructure='bcc', a=3.51, cubic=True)
    calc = get_calc(txt='elph_li.txt')
    atoms.calc = calc
    elph = DisplacementRunner(atoms, calc,
                              supercell=SUPERCELL, name='elph',
                              calculate_forces=True)
    elph.run()
    return elph


# supercell_cache should only be calculated once per session
@pytest.fixture(scope='session')
def supercell_cache(session_tmp_path, elph_cache):
    atoms = bulk('Li', crystalstructure='bcc', a=3.51, cubic=True)
    elph_cache
    calc = get_calc(parallel={'domain': 1, 'band': 1},
                    # parallel={'sl_auto': True, 'augment_grids':True,
                    #          'band': 2, 'kpt': 1, 'domain': 1 },
                    txt='gs_li.txt')

    # create supercell cache
    sc = Supercell(atoms, supercell=SUPERCELL)
    # use calc or calcdict
    # calcdict = dict(basis='sz(dzp)',
    #                 kpts={'size': (1, 2, 2), 'gamma': False},
    #                 txt='gs_li.txt')
    sc.calculate_supercell_matrix(calc)
    return sc
