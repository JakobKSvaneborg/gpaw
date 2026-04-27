def Reader(filename):
    import ase.io.ulm as ulm
    return ulm.Reader(filename)


def Writer(filename, world, mode='w', tag='GPAW'):
    import ase.io.ulm as ulm
    if world.rank == 0:
        return ulm.Writer(filename, mode=mode, tag=tag)
    return ulm.DummyWriter()
