from gpaw import GPAW_NO_C_EXTENSION

if GPAW_NO_C_EXTENSION:
    from gpaw.purepython import *  # noqa: F401, F403
else:
    from _gpaw import *  # noqa: F401, F403


def get_extension_module_path() -> str:
    """Return path to the _gpaw extension module. If the extension module is
    not available, returns path to a 'purepython' module that is used in
    place of the actual extension.
    """
    from pathlib import Path
    if GPAW_NO_C_EXTENSION:
        import gpaw.purepython as mod  # type:ignore
    else:
        import _gpaw as mod  # type:ignore
    return str(Path(mod.__file__).resolve())
