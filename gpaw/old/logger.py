import os
import sys
import time

import ase
import numpy as np
from ase import __version__ as ase_version
from ase.utils import IOContext, search_current_git_hash

import gpaw
import gpaw.cgpaw as cgpaw
from gpaw.utilities.memory import maxrss


class GPAWLogger:
    """Class for handling all text output."""
    def __init__(self, world):
        self.world = world

        self.verbose = False
        self._fd = None
        self.oldfd = 42
        self.iocontext = IOContext()
        self.use_colors = False
        self.green = ''
        self.reset = ''

    @property
    def fd(self):
        return self._fd

    @fd.setter
    def fd(self, fd):
        """Set the stream for text output.

        If `txt` is not a stream-object, then it must be one of:

        * None:  Throw output away.
        * '-':  Use stdout (``sys.stdout``) on master, elsewhere throw away.
        * A filename:  Open a new file on master, elsewhere throw away.
        """
        if fd == self.oldfd:
            return
        self.oldfd = fd
        self._fd = self.iocontext.openfile(fd, self.world)
        self.header()

    def __call__(self, *args, **kwargs):
        flush = kwargs.pop('flush', False)
        print(*args, file=self._fd, **kwargs)
        if flush and self._fd:
            self._fd.flush()

    def flush(self):
        self._fd.flush()

    def header(self):
        self()
        self('  ___ ___ ___ _ _ _  ')
        self(' |   |   |_  | | | | ')
        self(' | | | | | . | | | | ')
        self(' |__ |  _|___|_____| ', gpaw.__version__)
        self(' |___|_|             ')
        self()

        write_header(self, self.world)

    def print_dict(self, dct, sep='  '):
        options = np.get_printoptions()
        try:
            np.set_printoptions(threshold=4, linewidth=50)
            for key, value in sorted(dct.items()):
                if hasattr(value, 'todict'):
                    value = value.todict()
                if isinstance(value, dict):
                    sep = ',\n     ' + ' ' * len(key)
                    keys = sorted(value, key=lambda k: (str(type(k)), k))
                    s = sep.join(f'{k}: {value[k]}' for k in keys)
                    self(f'  {key}: {{{s}}}')
                elif hasattr(value, '__len__'):
                    value = np.asarray(value)
                    sep = ',\n    ' + ' ' * len(key)
                    s = sep.join(str(value).splitlines())
                    self(f'  {key}: {s}')
                else:
                    self(f'  {key}: {value}')
        finally:
            np.set_printoptions(**options)

    def __del__(self):
        """Destructor:  Write timing output before closing."""
        self.close()

    def close(self):
        if not self._fd:
            return
        if gpaw.dry_run or self._fd.closed:
            return

        try:
            mr = maxrss()
        except (LookupError, TypeError, NameError, AttributeError):
            # Thing can get weird during interpreter shutdown ...
            mr = 0

        if mr > 0:
            if mr < 1024**3:
                self('Memory usage: %.2f MiB' % (mr / 1024**2))
            else:
                self('Memory usage: %.2f GiB' % (mr / 1024**3))

        self('Date: ' + time.asctime())

        self.iocontext.close()

    def begin_table(self, title, header):
        pass


def write_header(log, world):
    # We use os.uname() here bacause platform.uname() starts a subprocess,
    # which MPI may not like!
    # This might not work on Windows.  We will see ...
    nodename, machine = os.uname()[1::3]

    log('User:  ', os.getenv('USER', '???') + '@' + nodename)
    log('Date:  ', time.asctime())
    log('Arch:  ', machine)
    log('Pid:   ', os.getpid())
    log('CWD:   ', os.getcwd())
    log('units:  Angstrom and eV')
    log('cores: ', world.size)
    log('OpenMP:', cgpaw.have_openmp)

    log('\nVersions:')
    log(f'  Python: {sys.version.split()[0]}')
    log(f'  ASE:    {ase_version}')
    log(f'  Numpy:  {np.version.version}')
    import scipy as sp
    log(f'  Scipy:  {sp.version.version}')
    log('  Libxc: ', getattr(cgpaw, 'libxc_version', '2.x.y'))

    log('\nPaths:')
    log('  gpaw: ', os.path.dirname(gpaw.__file__))
    line = os.path.normpath(cgpaw.get_extension_module_path())
    log('  _gpaw:', cut(line))
    log(f'  ase:   {os.path.dirname(ase.__file__)}')
    log(f'  numpy: {os.path.dirname(np.__file__)}')
    log(f'  scipy: {os.path.dirname(sp.__file__)}')
    # Explicitly deleting SciPy seems to remove garbage collection
    # problem of unknown cause
    del sp

    log('\nGit-hashes:')
    githash = search_current_git_hash(gpaw, world)
    if githash is not None:
        log(f'  gpaw:  {githash:.10}')
    if hasattr(cgpaw, 'githash'):
        log(f'  _gpaw: {cgpaw.githash():.10}')
    githash = search_current_git_hash(ase, world)
    if githash is not None:
        log(f'  ase:   {githash:.10}')

    log('\nOMP_NUM_THREADS:', os.environ.get('OMP_NUM_THREADS', ''))
    if gpaw.debug:
        log('DEBUG-MODE: true')
    log()


def cut(s, indent='        '):
    if len(s) + len(indent) < 80:
        return s
    s1, s2 = s.rsplit('/', 1)
    return s1 + '/\n' + indent + s2


def indent(s, level=1, tab='  '):
    return ''.join(tab * level + l for l in s.splitlines(True))
