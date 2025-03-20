from gpaw import GPAW_NO_C_EXTENSION

if GPAW_NO_C_EXTENSION:
    import gpaw.pp as _gpaw
    have_magma = False
else:
    import _gpaw  # type: ignore[no-redef]

    # Do not force users to recompile due to merging magma support to master
    have_magma = getattr(_gpaw, 'have_magma', False)


def __getattr__(name):
    return getattr(_gpaw, name)
