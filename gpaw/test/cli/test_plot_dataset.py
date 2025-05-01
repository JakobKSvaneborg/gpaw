import argparse
import contextlib
import os
import shlex
import subprocess
import sys

import pytest

from gpaw.setup_data import search_for_file
from gpaw.atom.plot_dataset import CLICommand, read_setup_file


setup_file = 'Ti.LDA'
installed_setup = read_setup_file(search_for_file(setup_file)[0])
if '=' in installed_setup.generatordata:
    full_fig_nplots = 4
    run_main_ctx = None
else:
    # Legacy setup files, no info on which to reconstruct the generator
    # -> cannot plot the log derivatives
    full_fig_nplots = 3
    run_main_ctx = pytest.warns(match='cannot reconstruct')


@pytest.mark.serial
@pytest.mark.parametrize(
    ('flags', 'search', 'use_cli', 'expected_nplots', 'ctx'),
    [('', False, False, 2, None),  # Minimal plot
     ('-s', False, False, 2, None),  # --separate-figures (ignored)
     ('-p -l spd,-1:1:.05',  # Reconstruct gen and make the full fig
      False, False, full_fig_nplots, run_main_ctx),
     ('-p -l spd,-1:1:.05',  # Same as the above, but...
      True, False,  # ... also test the dataset file searching
      full_fig_nplots, run_main_ctx),
     ('-p -l spd,-1:1:.05',  # Same as the above, but...
      True, True,  # ... also test the CLI
      full_fig_nplots, None)])  # Running in subproc, no warnings
def test_gpaw_plot_dataset(
        flags, search, use_cli, expected_nplots, ctx, in_tmp_dir):
    """
    Test for `gpaw plot-dataset`.
    """
    old_files = set(os.listdir(os.curdir))
    outfile = 'output.png'
    expected_files = {outfile}
    if search not in ('dataset', True):
        _, content = search_for_file(setup_file)
        with open(setup_file, mode='wb') as fobj:
            # Note: we can't directly use the file pointed to by
            # `search_for_file()` because it may be zipped
            fobj.write(content)
        expected_files.add(setup_file)

    if use_cli:
        argv = [sys.executable, '-m', 'gpaw', '-T', 'plot-dataset']
    else:
        argv = []
    argv.extend([f'--write={outfile}', *shlex.split(flags), setup_file])
    if search in (True,):
        argv.insert(-1, '--search')
        argv.insert(-1, '--')
    elif search:
        argv.insert(-1, '--search=' + search)
    if use_cli:
        subprocess.check_call(argv)
    else:
        parser = argparse.ArgumentParser()
        CLICommand.add_arguments(parser)
        with contextlib.nullcontext() if ctx is None else ctx:
            fig = CLICommand.run(parser.parse_args(argv))

    # Check output and inspect the figure where possible
    new_files = set(os.listdir(os.curdir))
    assert new_files == old_files | expected_files
    assert new_files - old_files == expected_files
    if use_cli:
        return
    ax_titles = [ax.get_title() for ax in fig.axes]
    assert len(ax_titles) == expected_nplots


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
