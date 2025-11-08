import sys
from pathlib import Path

import pytest

from gpaw import GPAW, PW


@pytest.mark.ci
def test_new_calculator(in_tmp_dir):
    """Test the GPAW.new() method."""

    params = dict(
        mode=PW(200),
        xc='LDA',
        nbands=8,
        kpts={'size': (4, 4, 4), 'gamma': True})

    modification_m = [
        dict(mode='fd'),
        dict(xc='PBE'),
        dict(nbands=10),
        dict(kpts={'size': (4, 4, 4)}),
        dict(kpts={'size': (3, 3, 3)}, xc='PBE')]

    calc0 = GPAW(**params, txt='calc0.txt')

    for m, modification in enumerate(modification_m):
        if m == 0:
            # Don't give a new txt file
            calc = calc0.new(**modification)
            check_file_handles(calc0, calc)
        else:
            txt = f'calc{m}.txt'
            calc = calc0.new(**modification, txt=txt)
            check_file_handles(calc0, calc, txt=txt)

        check_calc(calc, params, modification)


def check_file_handles(calc0, calc, txt=None):
    if calc.old:
        comm = calc.log.world
        assert comm.rank == calc0.log.world.rank
        fd = calc.log._fd
        fd0 = calc0.log._fd
    else:
        comm = calc.log.comm
        assert comm.rank == calc0.log.comm.rank
        fd = calc.log.fd
        fd0 = calc0.log.fd

    if comm.rank == 0:
        # We never want to reuse the output file
        assert fd is not fd0

        if txt is None:
            # When no txt is specified, the new calculator should log its
            # output in stdout
            assert fd is sys.stdout
        else:
            # Check that the new calculator log file handle was updated
            # appropriately
            assert Path(fd.name).name == txt


def check_calc(calc, params, modification):
    desired_params = params.copy()
    desired_params.update(modification)

    if calc.old:
        params = calc.parameters
    else:
        from gpaw.dft import Parameters
        desired_params = Parameters(**desired_params).todict()
        params = calc.params.todict()

    for param, value in desired_params.items():
        assert params[param] == value
