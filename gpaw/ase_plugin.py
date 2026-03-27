"""Plugin for ASE so that ASE can read log-file from a GPAW calculation."""
from ase.utils.plugins import ExternalIOFormat


io = ExternalIOFormat(
    desc='GPAW log-file',
    code='+F',  # +=multiple atoms objects, F=accepts a file-descriptor
    module='gpaw.ase_plugin')


def read_gpaw_log(filedesc, index):
    # (Just use the old reader from ASE for now)
    from ase.io.gpaw_out import read_gpaw_out
    return read_gpaw_out(filedesc, index)
