import pytest
from ase.build import bulk
from gpaw import GPAW
from gpaw.mixer import MixerFull
from gpaw.mpi import world


@pytest.mark.skipif(
    world.size > 4,
    reason='non-collinear calculation can only parallelize over k-points')
def test_co_new_mixing(gpaw_new):
    atoms = bulk('Co', crystalstructure='fcc', a=2.51 * 2**0.5)
    atoms.set_initial_magnetic_moments([2])
    kpts = (3, 3, 3)
    atoms.calc = GPAW(mode='pw',
                      kpts=kpts,
                      mixer=MixerFull())
    e1 = atoms.get_potential_energy()
    atoms.calc = GPAW(mode='pw',
                      kpts=kpts,
                      symmetry='off',
                      mixer=MixerFull(),
                      **(dict(magmoms=[[1, -1, 1]]) if gpaw_new else
                         dict(experimental=dict(magmoms=[[1, -1, 1]]))))
    e2 = atoms.get_potential_energy()
    assert e1 == pytest.approx(e2, abs=0.002)
