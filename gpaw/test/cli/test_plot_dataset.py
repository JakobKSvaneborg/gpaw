import argparse
import contextlib
import dataclasses
import os
import shlex
import subprocess
import sys
from typing import ContextManager

import pytest

from gpaw.setup_data import search_for_file
from gpaw.atom.plot_dataset import CLICommand, read_setup_file


@dataclasses.dataclass
class SetupInfo:
    filename: str
    nplots: int
    ctx: ContextManager = dataclasses.field(
        default_factory=contextlib.nullcontext)


@pytest.fixture(scope='module')
def old_setup():
    """
    Legacy setup file, no info on which to reconstruct the generator
    -> cannot plot the log derivatives, expect warnings
    """
    setup_file = 'Ti.LDA'
    installed_setup = read_setup_file(search_for_file(setup_file)[0])
    assert ('=' not in installed_setup.generatordata), (
        f'Setup {setup_file!r} has been updated to include more '
        'generator data, update the test')
    return SetupInfo(
        setup_file, 3, pytest.warns(match='cannot reconstruct'))


@pytest.fixture(scope='module')
def new_setup():
    """
    New setup file, reconstruction possible -> all plots can be plotted
    """
    setup_file = 'Cr.14.LDA'
    installed_setup = read_setup_file(search_for_file(setup_file)[0])
    assert ('=' in installed_setup.generatordata), (
        f'Setup {setup_file!r} does\'t have the expected generator data')
    return SetupInfo(setup_file, 4)


@pytest.mark.serial
@pytest.mark.parametrize('setup', ['old_setup', 'new_setup'])
@pytest.mark.parametrize(
    ('flags', 'search', 'use_cli', 'expected_nplots'),
    [('', False, False, 2),  # Minimal plot
     ('-s', False, False, 2),  # --separate-figures (ignored)
     ('-p -l spd,-1:1:.05',  # Reconstruct gen and make the full fig
      False, False, None),
     ('-p -l spd,-1:1:.05',  # Same as the above, but...
      True, False,  # ... also test the dataset file searching
      None),
     ('-p -l spd,-1:1:.05',  # Same as the above, but...
      True, True,  # ... also test the CLI
      None)])  # Running in subproc, no warnings
def test_gpaw_plot_dataset(
        setup, flags, search, use_cli, expected_nplots, request, in_tmp_dir):
    """
    Test for `gpaw plot-dataset`. For the cases where
    `expected_nplots = None`, try to make as many subplots as the
    dataset permits, potentially emitting a warning.
    """
    info = request.getfixturevalue(setup)
    setup_file = info.filename
    old_files = set(os.listdir(os.curdir))
    outfile = 'output.png'
    expected_files = {outfile}
    if expected_nplots is None:
        # Plot as many plots as possible, which may result in warnings
        # (to be caught by `info.ctx`)
        expected_nplots = info.nplots
        ctx = info.ctx
    else:
        # Plot a definite number of plots, expect no warnings
        ctx = contextlib.nullcontext()

    args = [f'--write={outfile}', *shlex.split(flags)]
    if search:
        args.append('--search')
    else:
        _, content = search_for_file(setup_file)
        with open(setup_file, mode='wb') as fobj:
            # Note: we can't directly use the file pointed to by
            # `search_for_file()` because it may be zipped
            fobj.write(content)
        expected_files.add(setup_file)
    args.append(setup_file)

    if use_cli:
        # Not much we can do about the subcommand, just see if it works
        gpaw_plot_cmd = [sys.executable, '-m', 'gpaw', '-T', 'plot-dataset']
        subprocess.check_call(gpaw_plot_cmd + args)
    else:
        parser = argparse.ArgumentParser()
        CLICommand.add_arguments(parser)
        with ctx:
            axs = CLICommand.run(parser.parse_args(args))
        # All plots should be on the same figure
        assert len({id(ax.get_figure()) for ax in axs}) == 1
        # Check we have as many plots as expected
        assert len(axs) == expected_nplots, (
            repr([ax.get_title() for ax in axs]))

    # Check existence of output file
    new_files = set(os.listdir(os.curdir))
    assert new_files == old_files | expected_files
    assert new_files - old_files == expected_files


@pytest.mark.serial
@pytest.mark.parametrize(('write', 'basis'),
                         [(False, False),  # Minimal
                          (True, True)])
def test_gpaw_dataset_plot(write, basis, in_tmp_dir):
    """
    Test for `gpaw dataset --plot`.
    """
    old_files = set(os.listdir(os.curdir))
    outfile = 'output.png'
    argv = [sys.executable, '-m', 'gpaw', '-T', 'dataset',
            f'--plot={outfile}', 'Ti']
    expected_files = {outfile}
    if write:
        argv.insert(-1, '-w')
        expected_files.add('Ti.LDA')
    if basis:
        argv.insert(-1, '-b')
        expected_files.add('Ti.dzp.basis')
    subprocess.check_call(argv)
    new_files = set(os.listdir(os.curdir))
    assert new_files == old_files | expected_files
    assert new_files - old_files == expected_files
