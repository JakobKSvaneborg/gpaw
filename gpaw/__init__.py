# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.
"""Main gpaw module."""
from __future__ import annotations
import os
import contextlib
from pathlib import Path
from typing import Any, TYPE_CHECKING
import warnings


__version__ = '25.7.1b1'
__ase_version_required__ = '3.25.0'

__all__ = ['GPAW',
           'Mixer', 'MixerSum', 'MixerDif', 'MixerSum2',
           'MixerFull',
           'CG', 'Davidson', 'RMMDIIS', 'DirectLCAO',
           'PoissonSolver',
           'FermiDirac', 'MethfesselPaxton', 'MarzariVanderbilt',
           'PW', 'LCAO', 'FD',
           'restart']

boolean_envvars = {
    'GPAW_NEW',
    'GPAW_CPUPY',
    'GPAW_USE_GPUS',
    'GPAW_TRACE',
    'GPAW_NO_C_EXTENSION',
    'GPAW_MPI4PY',
    'GPAW_DEBUG',
    'GPAW_NO_GPU_MPI'}
allowed_envvars = {
    *boolean_envvars,
    'GPAW_MPI_OPTIONS',
    'GPAW_MPI',
    'GPAW_SETUP_PATH'}

dry_run = 0


def _get_gpaw_env_vars(attr: str) -> bool | str:
    if attr in boolean_envvars:
        return bool(int(os.environ.get(attr) or 0))
    if attr in allowed_envvars and attr in os.environ:
        return os.environ[attr]
    raise _module_attr_error(attr)


# When type-checking, we want the debug-wrappers enabled:
debug = TYPE_CHECKING or _get_gpaw_env_vars('GPAW_DEBUG')

# Debug envvar for disabling GPU aware MPI
ENVVAR_GPAW_NO_GPU_MPI = _get_gpaw_env_vars('GPAW_NO_GPU_MPI')


@contextlib.contextmanager
def disable_dry_run():
    """Context manager for temporarily disabling dry-run mode.

    Useful for skipping exit in the GPAW constructor.
    """
    global dry_run
    size = dry_run
    dry_run = 0
    yield
    dry_run = size


def get_scipy_version():
    import scipy
    # This is in a function because we don't like to have the scipy
    # import at module level
    return [int(x) for x in scipy.__version__.split('.')[:2]]


if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '1'


class ConvergenceError(Exception):
    pass


class KohnShamConvergenceError(ConvergenceError):
    pass


class PoissonConvergenceError(ConvergenceError):
    pass


class KPointError(Exception):
    pass


class BadParallelization(Exception):
    """Error indicating missing parallelization support."""
    pass


def get_libraries() -> dict[str, str]:
    import gpaw.cgpaw as cgpaw
    libraries: dict[str, str] = {}
    if hasattr(cgpaw, 'lxcXCFunctional'):
        libraries['libxc'] = getattr(cgpaw, 'libxc_version', '2.x.y')
    else:
        libraries['libxc'] = ''
    return libraries


def __getattr__(attr: str) -> Any:
    for attr_getter in _lazy_import, _get_gpaw_env_vars:
        try:
            result = attr_getter(attr)
        except AttributeError:
            continue
        return globals().setdefault(attr, result)
    raise _module_attr_error(attr)


def __dir__() -> list[str]:
    """
    Get the (1) normally-present module attributes, (2) lazily-imported
    objects, and (3) envrionmental variables starting with `GPAW_`.
    """
    return list({*globals(),
                 *all_lazy_imports,  # From `_lazy_import()`
                 *{*boolean_envvars,  # From `_get_gpaw_env_vars()`
                   *(var for var in os.environ if var.startswith('GPAW_'))}})


def _module_attr_error(attr: str, *args, **kwargs) -> AttributeError:
    return AttributeError(f'{__getattr__.__module__}: '
                          f'no attribute named `.{attr}`',
                          *args, **kwargs)


def _lazy_import(attr: str) -> Any:
    """
    Implement the lazy importing of classes in submodules."""
    import importlib

    try:
        import_target = all_lazy_imports[attr]
    except KeyError:
        raise _module_attr_error(attr) from None

    module, sep, target = import_target.rpartition('.')
    assert module and all(chunk.isidentifier() for chunk in module.split('.'))
    assert sep
    assert target.isidentifier()
    return getattr(importlib.import_module(module), target)


all_lazy_imports = dict(
    Mixer='gpaw.mixer.Mixer',
    MixerSum='gpaw.mixer.MixerSum',
    MixerDif='gpaw.mixer.MixerDif',
    MixerSum2='gpaw.mixer.MixerSum2',
    MixerFull='gpaw.mixer.MixerFull',

    Davidson='gpaw.eigensolvers.Davidson',
    RMMDIIS='gpaw.eigensolvers.RMMDIIS',
    CG='gpaw.eigensolvers.CG',
    DirectLCAO='gpaw.eigensolvers.DirectLCAO',

    PoissonSolver='gpaw.poisson.PoissonSolver',
    FermiDirac='gpaw.occupations.FermiDirac',
    MethfesselPaxton='gpaw.occupations.MethfesselPaxton',
    MarzariVanderbilt='gpaw.occupations.MarzariVanderbilt',
    FD='gpaw.old.wavefunctions.fd.FD',
    LCAO='gpaw.old.wavefunctions.lcao.LCAO',
    PW='gpaw.old.wavefunctions.pw.PW')


# Make sure e.g. GPAW_NEW=0 will set GPAW_NEW=False
# (`__getattr__()` magic handles the other boolean environment
# variables, but GPAW_NEW is used within the same script, so it needs to
# concretely exist in the namespace)
GPAW_NEW = _get_gpaw_env_vars('GPAW_NEW')

if os.uname().machine == 'wasm32':
    GPAW_NO_C_EXTENSION = True


class BroadcastImports:
    def __enter__(self):
        from gpaw._broadcast_imports import broadcast_imports
        self._context = broadcast_imports
        return self._context.__enter__()

    def __exit__(self, *args):
        self._context.__exit__(*args)


broadcast_imports = BroadcastImports()


if debug:
    import numpy as np
    np.seterr(over='raise', divide='raise', invalid='raise', under='ignore')
    oldempty = np.empty
    oldempty_like = np.empty_like

    def empty(*args, **kwargs):
        a = oldempty(*args, **kwargs)
        try:
            a.fill(np.nan)
        except ValueError:
            a.fill(42)
        return a

    def empty_like(*args, **kwargs):
        a = oldempty_like(*args, **kwargs)
        try:
            a.fill(np.nan)
        except ValueError:
            a.fill(-42)
        return a

    np.empty = empty  # type: ignore[misc]
    np.empty_like = empty_like

if TYPE_CHECKING:
    from gpaw.new.ase_interface import GPAW
elif GPAW_NEW:
    all_lazy_imports['GPAW'] = 'gpaw.dft.GPAW'
else:
    all_lazy_imports['GPAW'] = 'gpaw.old.calculator.GPAW'

all_lazy_imports['get_calculation_info'] = 'gpaw.calcinfo.get_calculation_info'


def restart(filename, Class=None, **kwargs):
    if Class is None:
        from gpaw import GPAW as Class
    calc = Class(filename, **kwargs)
    atoms = calc.get_atoms()
    return atoms, calc


def read_rc_file():
    home = os.environ.get('HOME')
    if home is not None:
        rc = os.path.join(home, '.gpaw', 'rc.py')
        if os.path.isfile(rc):
            # Read file in ~/.gpaw/rc.py
            with open(rc) as fd:
                exec(fd.read())


def initialize_data_paths():
    try:
        setup_paths[:0] = os.environ['GPAW_SETUP_PATH'].split(os.pathsep)
    except KeyError:
        pass


def standard_setup_paths() -> list[str | Path]:
    try:
        import gpaw_data
    except ModuleNotFoundError:
        return []
    else:
        return [gpaw_data.datapath()]


setup_paths = standard_setup_paths()
read_rc_file()
initialize_data_paths()


def RMM_DIIS(*args, **kwargs):
    from gpaw import RMMDIIS
    warnings.warn('Please use RMMDIIS instead of RMM_DIIS')
    return RMMDIIS(*args, **kwargs)
