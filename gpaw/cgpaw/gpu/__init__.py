from gpaw import GPAW_NO_C_EXTENSION

_ok = False
if not GPAW_NO_C_EXTENSION:
    try:
        from _gpaw.gpu import *  # noqa: F401, F403
        _ok = True
    except (ImportError, ModuleNotFoundError):
        pass


if not _ok:
    # ... purepython stuff ...
    pass
