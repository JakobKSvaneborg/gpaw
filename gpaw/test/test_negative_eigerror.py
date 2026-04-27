from ase.spacegroup import crystal

from gpaw import PW, MethfesselPaxton


def test_negative_eigerror(mpi):
    graphite = crystal(symbols=['C', 'C'],
                       basis=[[0, 0, 1 / 4],
                              [1 / 3, 2 / 3, 1 / 4]],
                       spacegroup='P63/mmc',
                       cellpar=[2.464, 2.464, 6.711, 90, 90, 120])

    graphite.calc = mpi.GPAW(
        mode=PW(400),
        occupations=MethfesselPaxton(5.0, 1),
        nbands=20,
        kpts=(3, 3, 3))

    graphite.get_potential_energy()
