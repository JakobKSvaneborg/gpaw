from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path


class CachedFilesHandler(ABC):
    """Base class for objects which handle the writing
    and caching of session-scoped pytest fixtures"""

    def __init__(self, folder: Path, outformat: str, comm):
        self.folder = folder
        self.outformat = outformat
        self.comm = comm

        self.cached_files = {}
        for file in folder.glob('*' + outformat):
            self.cached_files[file.name[:-len(outformat)]] = file

    def __getitem__(self, name: str) -> Path:
        if name in self.cached_files:
            return self.cached_files[name]

        filepath = self.folder / (name + self.outformat)

        lockfile = self.folder / f'{name}.lock'

        comm = self.comm
        for _attempt in range(60):  # ~60s timeout
            file_exist = 0
            if comm.rank == 0:
                file_exist = int(filepath.exists())
            file_exist = comm.sum_scalar(file_exist)

            if file_exist:
                self.cached_files[name] = filepath
                return self.cached_files[name]

            try:
                with _temporary_lock(lockfile, comm):
                    work_path = filepath.with_suffix('.tmp' + self.outformat)
                    self._calculate_and_write(name, work_path)

                    # By now files should exist *and* be fully written, by us.
                    # Rename them to the final intended paths:
                    if comm.rank == 0:
                        work_path.rename(filepath)

            except Locked:
                import time
                time.sleep(1)

        raise RuntimeError(f'{self.__class__.__name__} fixture generation '
                           f'takes too long: {name}.  Consider using pytest '
                           '--cache-clear if there are stale lockfiles, '
                           'else write faster tests.')

    @abstractmethod
    def _calculate_and_write(self, name, work_path):
        """Get name from self and write to work_path."""


class Locked(FileExistsError):
    pass


@contextmanager
def _temporary_lock(path, comm):
    if comm.rank == 0:
        try:
            with _file_lock(path):
                comm.sum_scalar(1)
                yield
        except Locked:
            comm.sum_scalar(0)
            raise
    else:
        status = comm.sum_scalar(0)
        if status:
            yield
        else:
            raise Locked


@contextmanager
def _file_lock(path):
    fd = None
    try:
        with path.open('x') as fd:
            yield
    except FileExistsError:
        raise Locked()
    finally:
        if fd is not None:
            path.unlink()
