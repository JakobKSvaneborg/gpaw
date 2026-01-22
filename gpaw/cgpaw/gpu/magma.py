have_magma = False

try:
    from _gpaw.gpu.magma import *  # noqa: F401, F403
    have_magma = True
except ImportError:
    pass
