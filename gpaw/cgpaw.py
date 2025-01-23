from gpaw import no_c_extension

if no_c_extension:
    import gpaw.pp as _gpaw
else:
    import _gpaw


def __getattr__(name):
    return getattr(_gpaw, name)
