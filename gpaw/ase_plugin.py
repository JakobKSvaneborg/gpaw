from ase.utils.plugins import ExternalIOFormat


io = ExternalIOFormat(
    desc='GPAW log-file',
    code='+F',
    module='gpaw.plugin')


def read_gpaw_log(filename, index):
    asdgkjh
