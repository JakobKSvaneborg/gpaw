import numpy as np
import pytest
from ase import Atoms

from gpaw import GPAW
from gpaw.mpi import ibarrier, world


@pytest.mark.ci
def test_fixdensity(in_tmp_dir, gpaw_new, mpi):
    a = 2.5
    slab = Atoms('Li', cell=(a, a, 2 * a), pbc=1)
    slab.calc = mpi.GPAW(
        mode='fd', kpts=(3, 3, 1), txt='li-1.txt',
        nbands=5,
        parallel=dict(kpt=1))
    slab.get_potential_energy()
    slab.calc.write('li.gpw')

    # Gamma point:
    e1 = slab.calc.get_eigenvalues(kpt=0)[0]
    f1 = slab.calc.get_fermi_level()

    kpts = [(0, 0, 0)]

    # Fix density and continue:
    calc = slab.calc.fixed_density(
        txt='li-2.txt',
        nbands=5,
        kpts=kpts)
    e2 = calc.get_eigenvalues(kpt=0)[0]
    f2 = calc.get_fermi_level()

    # Start from gpw-file:
    calc = mpi.GPAW('li.gpw')
    calc = calc.fixed_density(
        txt='li-3.txt',
        nbands=5,
        kpts=kpts)
    e3 = calc.get_eigenvalues(kpt=0)[0]
    f3 = calc.get_fermi_level()
    assert f2 == pytest.approx(f1, abs=1e-10)
    assert f3 == pytest.approx(f1, abs=1e-10)
    assert e2 == pytest.approx(e1, abs=3e-5)
    assert e3 == pytest.approx(e1, abs=3e-5)
    o3 = calc.get_occupation_numbers(kpt=0, raw=True)[0]
    assert o3 == pytest.approx(1.0)


@pytest.mark.ci
@pytest.mark.skipif(world.size == 1, reason='only parallel')
def test_fixdensity_world(in_tmp_dir, mpi):
    a = 2.5
    slab = Atoms('H', cell=(a, a, a), pbc=1)
    comm = mpi.comm.new_communicator(range(world.size // 2))
    if not comm:
        # Don't actually hang, if this fails
        ibarrier(timeout=10, comm=mpi.comm)
        return
    slab.calc = GPAW(mode='pw', kpts=(1, 1, 1), txt='H.txt',
                     communicator=comm)
    slab.get_potential_energy()
    slab.calc.write('li.gpw')
    e1 = slab.calc.get_eigenvalues(kpt=0)

    # Fix density and continue:
    calc = slab.calc.fixed_density(
        txt='H2.txt',
        kpts={'gamma': True, 'size': [2, 2, 2]})
    e2 = calc.get_eigenvalues(kpt=0)
    assert np.allclose(e1, e2, atol=1e-2)
    ibarrier(timeout=10, comm=mpi.comm)
