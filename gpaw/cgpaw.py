from gpaw import GPAW_NO_C_EXTENSION

if GPAW_NO_C_EXTENSION:
    have_magma = False
    import gpaw.purepython as _gpaw
else:
    import _gpaw  # type: ignore[no-redef]

try:
    import _gpaw.gpu.magma
    have_magma = True
except ImportError:
    have_magma = False


def __getattr__(name):
    return getattr(_gpaw, name)
