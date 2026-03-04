from gpaw import GPAW_NO_C_EXTENSION


_ok = True
if not GPAW_NO_C_EXTENSION:
    try:
        from _gpaw.gpu import *  # noqa: F401, F403
    except (ImportError, ModuleNotFoundError):
        _ok = False


if not GPAW_NO_C_EXTENSION or not _ok:
    # ... purepython stuff ...
    pass
