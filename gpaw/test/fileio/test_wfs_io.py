from ase import Atoms


def test_fileio_wfs_io(in_tmp_dir, mpi):
    h2 = Atoms('H2', [(0, 0, 0), (0, 0, 1)])
    h2.center(vacuum=2.0)
    calc = mpi.GPAW(mode='fd', nbands=2, convergence={'eigenstates': 1e-3})
    h2.calc = calc
    calc.create_new_calculation(h2)
    for ctx in calc.dft.iconverge():
        pass
    r0 = ctx.wfs.eigensolver.error
    assert r0 < 1e-3
    calc.write('h2', 'all')

    # refine the restart file containing the wfs
    calc = mpi.GPAW('h2', convergence={'eigenstates': 1e-5})
    for ctx in calc.dft.iconverge():
        pass
    r1 = ctx.wfs.eigensolver.error
    assert r1 < 1e-5
