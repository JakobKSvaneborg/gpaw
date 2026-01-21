from gpaw import GPAW_NO_C_EXTENSION
from types import ModuleType

# Hack?? alias all submodules of _gpaw for the import system
import sys

if GPAW_NO_C_EXTENSION:
    from gpaw.purepython import *
else:
    from _gpaw import *
    import _gpaw

    sys.modules["gpaw.cgpaw.gpu"] = _gpaw.gpu

    try:
        import _gpaw.gpu.magma  # type: ignore[no-redef]
        have_magma = True
        sys.modules["gpaw.cgpaw.gpu.magma"] = _gpaw.gpu.magma
    except ImportError:
        have_magma = False


def _get_extension_module() -> ModuleType | None:
    """Return direct access to the _gpaw extension module, if available.
    Internal use only.
    """
    if GPAW_NO_C_EXTENSION:
        return None
    else:
        return _gpaw
