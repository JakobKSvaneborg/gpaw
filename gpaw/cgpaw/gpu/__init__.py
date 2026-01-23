try:
    from _gpaw.gpu import *  # noqa: F401, F403
except (ImportError, ModuleNotFoundError):
    pass
