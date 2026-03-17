import pytest

from ase.build import bulk
from gpaw.dft import Symmetry


def noncontiguous_symmetries(atoms):
    symm = Symmetry().build(atoms)
    return Symmetry(
        rotations=symm.rotation_scc.transpose(0, 2, 1),
        translations=symm.translation_sc,
        atommaps=symm.atommap_sa)


@pytest.mark.serial  # will only raise on rank-0
def test_noncontiguous(mpi):
    """Test that we raise an error if rotation array is not contiguous."""

    atoms = bulk('Au')

    atoms.calc = mpi.NewGPAW(
        mode='pw',
        symmetry=noncontiguous_symmetries(atoms))

    with pytest.raises(TypeError, match='Not a proper NumPy array'):
        atoms.get_potential_energy()
