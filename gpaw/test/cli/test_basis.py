"""
Test the `gpaw basis` subcommand.
"""
from __future__ import annotations

import pathlib
import subprocess
import shlex
import sys
from collections.abc import Collection

import pytest

from gpaw.setup_data import search_for_file


@pytest.mark.parametrize(
    ('search', 'flags', 'expected_outputs'),
    [
        # Standard use
        (False, None, ['H.dzp.basis', 'Cl.dzp.basis']),
        # Search for the setup files among the installed datasets
        (True, None, ['H.dzp.basis', 'Cl.dzp.basis']),
        # Triple-zeta polarized functions
        (True, '--type=tzp', ['H.tzp.basis', 'Cl.tzp.basis']),
        # Other flags
        (True,
         '--name=foo --tail=.1,.2,.3 --energy=.3 --vconf-sharp-confinement',
         ['H.foo.dzp.basis', 'Cl.foo.dzp.basis'])])
@pytest.mark.serial
def test_gpaw_basis_subcommand(
        search: bool,
        flags: str | None,
        expected_outputs: Collection[str],
        in_tmp_dir: pathlib.Path) -> None:
    """
    Test generating basis-set files with the `gpaw basis` subcommand.
    """
    cmd = [sys.executable, '-m', 'gpaw', '-T', 'basis']
    setup_files = ['H.PBE', 'Cl.PBE']
    if flags is not None:
        cmd.extend(shlex.split(flags))
    if search:
        cmd.append('--search')
        cmd.extend(setup_files)
    else:
        for setup in setup_files:
            filename, _ = search_for_file(setup)
            cmd.append(filename)
    assert not subprocess.run(cmd).returncode
    files = {path.name for path in in_tmp_dir.iterdir()}
    assert files == set(expected_outputs)


@pytest.mark.serial
def test_old_gpaw_basis_tool(
        in_tmp_dir: pathlib.Path) -> None:
    """
    Test generating basis-set files with the old `gpaw-basis` CLI tool.
    """
    # From atomic species
    cmd = ['gpaw-basis',
           '--xc', 'PBE',
           '--name', 'myfile',
           '--type', 'tzp',
           '--save-setup',
           'H', 'Cl']
    assert not subprocess.run(cmd).returncode
    assert ({path.name for path in in_tmp_dir.iterdir()}
            == {'H.myfile.tzp.basis', 'Cl.myfile.tzp.basis',
                'H.myfile.PBE', 'Cl.myfile.PBE'})
    # From the generated PAW setups
    assert not subprocess.run(['gpaw-basis',
                               'H.myfile.PBE', 'Cl.myfile.PBE']).returncode
    assert ({path.name for path in in_tmp_dir.iterdir()}
            == {'H.myfile.tzp.basis', 'Cl.myfile.tzp.basis',
                'H.myfile.PBE', 'Cl.myfile.PBE',
                'H.dzp.basis', 'Cl.dzp.basis'})
