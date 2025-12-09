from ase import Atoms
from ase.units import Hartree

from gpaw import GPAW


def test_fileio_wfs_io(in_tmp_dir):
    h2 = Atoms('H2', [(0, 0, 0), (0, 0, 1)])
    h2.center(vacuum=2.0)
    calc = GPAW(mode='fd', nbands=2, convergence={'eigenstates': 1e-3})
    h2.calc = calc
    if calc.old:
        h2.get_potential_energy()
        r0 = calc.wfs.eigensolver.error * Hartree**2 / 2
    else:
        calc.create_new_calculation(h2)
        for ctx in calc.dft.iconverge():
            pass
        r0 = ctx.wfs.eigensolver.error
    assert r0 < 1e-3
    calc.write('h2', 'all')

    # refine the restart file containing the wfs
    calc = GPAW('h2', convergence={'eigenstates': 1e-5})
    if calc.old:
        calc.get_atoms().get_potential_energy()
        r1 = calc.wfs.eigensolver.error * Hartree**2 / 2
    else:
        for ctx in calc.dft.iconverge():
            pass
        r1 = ctx.wfs.eigensolver.error
    assert r1 < 1e-5
