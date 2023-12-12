import pytest


@pytest.mark.serial
@pytest.mark.wannier
def test_ase_features_wannierk(in_tmp_dir, gpw_files):
    'Test ase.dft.wannier module with k-points.'
    from ase.build import bulk
    from ase.dft.wannier import Wannier

    from gpaw import GPAW
    from gpaw.mpi import world

    si = bulk('Si', 'diamond', a=5.43)
    k = 3

    def wan(calc):
        centers = [([0.125, 0.125, 0.125], 0, 1.5),
                   ([0.125, 0.625, 0.125], 0, 1.5),
                   ([0.125, 0.125, 0.625], 0, 1.5),
                   ([0.625, 0.125, 0.125], 0, 1.5)]
        w = Wannier(4, calc,
                    nbands=4,
                    # log=print,
                    initialwannier=centers)
        w.localize()
        x = w.get_functional_value()
        centers = (w.get_centers(1) * k) % 1
        c = (centers - 0.125) * 2
        print(w.get_radii())  # broken! XXX
        assert abs(c.round() - c).max() < 0.03
        c = sorted(c.round().astype(int).tolist())
        assert c == [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]
        if 0:
            from ase.visualize import view
            from ase import Atoms
            watoms = calc.atoms + Atoms(symbols='X4',
                                        scaled_positions=centers,
                                        cell=calc.atoms.cell)
            view(watoms)
        return x

    calc1 = GPAW(gpw_files['si_fd_bz'])
    x1 = wan(calc1)
    calc2 = GPAW(gpw_files['si_fd_ibz'])
    calc2.wfs.ibz2bz(si)
    x2 = wan(calc2)
    if world.rank == 0:
        print((x1, x2))
    assert abs(x1 - x2) < 0.001
    assert abs(x1 - 8.817) < 0.01

    world.barrier()
