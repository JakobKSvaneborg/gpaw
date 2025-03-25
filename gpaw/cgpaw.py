from gpaw import GPAW_NO_C_EXTENSION

if GPAW_NO_C_EXTENSION:
    import gpaw.purepython as _gpaw
else:
    import _gpaw  # type: ignore[no-redef]


def __getattr__(name):
    return getattr(_gpaw, name)
