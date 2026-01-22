try:
    from _gpaw.gpu.magma import *  # noqa: F401, F403
except ImportError:
    raise ImportError("MAGMA not available")
