from gpaw import GPAW_NO_C_EXTENSION
from types import ModuleType

if GPAW_NO_C_EXTENSION:
    from gpaw.purepython import *
else:
    from _gpaw import *

try:
    import _gpaw.gpu.magma  # type: ignore[no-redef]
    have_magma = True
except ImportError:
    have_magma = False


def _get_extension_module() -> ModuleType | None:
    """Return direct access to the _gpaw extension module, if available.
    Internal use only.
    """
    if GPAW_NO_C_EXTENSION:
        return None
    else:
        import _gpaw
        return _gpaw