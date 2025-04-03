import os
import shlex
import subprocess
import sys

import pytest

from gpaw.setup_data import search_for_file


@pytest.fixture(autouse=True)
def use_tmp_path(tmp_path):
    curdir = os.path.abspath(os.curdir)
    os.chdir(tmp_path)
    yield
    os.chdir(curdir)


@pytest.mark.parametrize(('flags', 'basis'),
                         [('', False),  # Minimal plot
                          ('-s', False),  # --separate-figures (ignored)
                          ('-p -l spd,-1:1:.05',  # Reconstruct gen
                           True)])  # Also load and plot the basis
def test_gpaw_plot_dataset(flags, basis):
    """
    Test for `gpaw plot-dataset`.
    """
    old_files = set(os.listdir(os.curdir))
    _, content = search_for_file('Ti.LDA')
    setup_file = 'Ti.LDA'
    with open(setup_file, mode='wb') as fobj:
        # Note: we can't directly use the file pointed to by
        # `search_for_file()` because it may be zipped
        fobj.write(content)
    outfile = 'output.png'
    expected_files = {outfile, setup_file}
    argv = [sys.executable, '-m', 'gpaw', '-T', 'plot-dataset',
            f'--write={outfile}', *shlex.split(flags), setup_file]
    if basis:
        _, content = search_for_file('Ti.dzp.basis')
        basis_file = 'T.dzp.basis'
        with open(basis_file, mode='wb') as fobj:
            fobj.write(content)
        argv.insert(-1, '-b' + basis_file)
        expected_files.add(basis_file)
    subprocess.check_call(argv)
    new_files = set(os.listdir(os.curdir))
    assert new_files == old_files | expected_files
    assert new_files - old_files == expected_files


@pytest.mark.parametrize(('write', 'basis'),
                         [(False, False),  # Minimal
                          (True, True)])
def test_gpaw_dataset_plot(write, basis):
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
