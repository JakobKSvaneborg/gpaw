from gpaw import GPAW_NO_C_EXTENSION

_ok = True

if not GPAW_NO_C_EXTENSION:

    try:
        from _gpaw.gpu.magma import *  # noqa: F401, F403
    except (ImportError, ModuleNotFoundError):
        _ok = False


if GPAW_NO_C_EXTENSION or not _ok:

    def available() -> bool:
        return False
