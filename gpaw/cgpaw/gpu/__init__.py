from gpaw import GPAW_NO_C_EXTENSION

if GPAW_NO_C_EXTENSION:
    raise ImportError("No purepython versions of GPU extensions")
else:
    from _gpaw.gpu import *  # noqa: F401, F403
