try:
    from _gpaw.gpu.magma import *
except ImportError:
    raise ImportError("MAGMA not available")
