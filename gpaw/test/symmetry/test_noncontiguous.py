import pytest

from ase.build import bulk
from gpaw.dft import Symmetry


def noncontiguous_symmetries(atoms):
    symm = Symmetry().build(atoms)
    symm.rotation_scc = symm.rotation_scc.transpose(0, 2, 1)
    return Symmetry(
        rotations=symm.rotation_scc,
        translations=symm.translation_sc,
        atommaps=symm.atommap_sa)


def test_noncontiguous(mpi):
    atoms = bulk('Au')

    atoms.calc = mpi.NewGPAW(
        mode='pw',
        symmetry=noncontiguous_symmetries(atoms))

    with pytest.raises(TypeError, match='Not a proper NumPy array'):
        atoms.get_potential_energy()
