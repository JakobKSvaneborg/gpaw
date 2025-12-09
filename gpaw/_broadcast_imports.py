"""Provide mechanism to broadcast imports from master to other processes.

This reduces file system strain.

Use:

  with broadcast_imports():
      <execute import statements>

This temporarily overrides the Python import mechanism so that

  1) master executes and caches import metadata and code
  2) import metadata and code are broadcast to all processes
  3) other processes execute the import statements from memory

Warning: Do not perform any parallel operations while broadcast imports
are enabled.  Non-master processes assume that they will receive module
data and will crash or deadlock if master sends anything else.
"""


import marshal
import os
import sys
from importlib.machinery import ModuleSpec, PathFinder

import gpaw.cgpaw as cgpaw
from gpaw import GPAW_INITIALIZE_MPI, GPAW_MPI_BACKEND, GPAW_NO_C_EXTENSION

cgpaw_version = getattr(cgpaw, 'version', 0)
if not GPAW_NO_C_EXTENSION and cgpaw_version != 10:
    improvement = ''
    if cgpaw_version == 9:
        improvement = ('GPAW has now much reduced memory consumption due to '
                       'optimized pwlfc_expand function in new GPAW. Enjoy. ')

    raise ImportError(improvement + 'Please recompile GPAW''s C-extensions!')


def init_mpi4py():
    from mpi4py.MPI import COMM_WORLD

    from gpaw.mpi4pywrapper import MPI4PYWrapper
    return MPI4PYWrapper(COMM_WORLD)


def init_cgpaw():
    libmpi = os.environ.get('GPAW_MPI', 'libmpi.so')
    import ctypes
    try:
        ctypes.CDLL(libmpi, ctypes.RTLD_GLOBAL)
    except OSError:
        pass
    return cgpaw.Communicator()


if GPAW_INITIALIZE_MPI:
    if GPAW_MPI_BACKEND == 'mpi4py':
        world = init_mpi4py()
    elif GPAW_MPI_BACKEND == 'cgpaw':
        if hasattr(cgpaw, 'Communicator'):
            world = init_cgpaw()
        else:
            # Would be cleaner for this to be an error since we are not
            # quite obeying the envvar.
            world = None  # type: ignore
    elif GPAW_MPI_BACKEND == 'serial':
        world = None  # type: ignore
    else:
        raise ValueError(
            "GPAW_MPI_BACKEND must be one of 'serial', 'cgpaw', 'mpi4py'")
else:
    world = None  # type: ignore


if world is None and 'OMPI_COMM_WORLD_SIZE' in os.environ:
    # Check whether we might not have the same ideas about parallelism
    # as the caller.
    #
    # This check is not portable to other MPIs.  Maybe we can have this
    # sanity check for a few MPI implementations since it's nasty to get
    # inconsistent MPI communicators.
    raise RuntimeError(
        'We appear to be running inside mpiexec, but '
        'parallelism is disabled.  Please run '
        'gpaw -P <nprocs> python to ensure that MPI is enabled.')


def marshal_broadcast(obj):
    if world.rank == 0:
        buf = marshal.dumps(obj)
    else:
        assert obj is None
        buf = None

    buf = cgpaw.globally_broadcast_bytes(buf)
    try:
        return marshal.loads(buf)
    except ValueError as err:
        msg = ('Parallel import failure -- probably received garbage.  '
               'Error was: {}.  This may happen if parallel operations are '
               'performed while parallel imports are enabled.'.format(err))
        raise ImportError(msg)


class BroadcastLoader:
    def __init__(self, spec, module_cache):
        self.module_cache = module_cache
        self.spec = spec

    def create_module(self, spec):
        # Returning None means to create the (uninitialized) module
        # in the same way as normal.
        #
        # (But we could return e.g. a subclass of Module if we wanted.)
        return None

    def exec_module(self, module):
        if world.rank == 0:
            # Load from file and store in cache:
            code = self.spec.loader.get_code(module.__name__)
            metadata = (self.spec.submodule_search_locations, self.spec.origin)
            self.module_cache[module.__name__] = (metadata, code)
            # We could execute the default mechanism to load the module here.
            # Instead we load from cache using our own loader, like on the
            # other cores.

        return self.load_from_cache(module)

    def load_from_cache(self, module):
        metadata, code = self.module_cache[module.__name__]
        origin = metadata[1]
        module.__file__ = origin
        # __package__, __path__, __cached__?
        module.__loader__ = self
        sys.modules[module.__name__] = module
        exec(code, module.__dict__)
        return module

    def __str__(self):
        return ('<{} for {}:{} [{} modules cached]>'
                .format(self.__class__.__name__,
                        self.spec.name, self.spec.origin,
                        len(self.module_cache)))


class BroadcastImporter:
    def __init__(self):
        self.module_cache = {}
        self.cached_modules = []

    def find_spec(self, fullname, path=None, target=None):
        if world.rank == 0:
            spec = PathFinder.find_spec(fullname, path, target)
            if spec is None:
                return None

            if spec.loader is None:
                return None

            code = spec.loader.get_code(fullname)
            if code is None:  # C extensions
                return None

            loader = BroadcastLoader(spec, self.module_cache)
            assert fullname == spec.name

            searchloc = spec.submodule_search_locations
            spec = ModuleSpec(fullname, loader, origin=spec.origin,
                              is_package=searchloc is not None)
            if searchloc is not None:
                spec.submodule_search_locations += searchloc
            return spec
        else:
            if fullname not in self.module_cache:
                # Could this in principle interfere with builtin imports?
                return PathFinder.find_spec(fullname, path, target)

            searchloc, origin = self.module_cache[fullname][0]
            loader = BroadcastLoader(None, self.module_cache)
            spec = ModuleSpec(fullname, loader, origin=origin,
                              is_package=searchloc is not None)
            if searchloc is not None:
                spec.submodule_search_locations += searchloc
            loader.spec = spec  # XXX loader.loader is still None
            return spec

    def broadcast(self):
        if world.size == 1:
            return
        if world.rank == 0:
            # print('bcast {} modules'.format(len(self.module_cache)))
            marshal_broadcast(self.module_cache)
        else:
            self.module_cache = marshal_broadcast(None)
            # print('recv {} modules'.format(len(self.module_cache)))

    def enable(self):
        if world is None:
            return

        # There is the question of whether we lose anything by inserting
        # ourselves further on in the meta_path list.  Maybe not, and maybe
        # that is a less violent act.
        sys.meta_path.insert(0, self)
        if world.rank != 0:
            self.broadcast()

    def disable(self):
        if world is None:
            return

        if world.rank == 0:
            self.broadcast()
        self.cached_modules += self.module_cache.keys()
        self.module_cache = {}
        myself = sys.meta_path.pop(0)
        assert myself is self

    def __enter__(self):
        self.enable()

    def __exit__(self, *args):
        self.disable()


broadcast_imports = BroadcastImporter()
