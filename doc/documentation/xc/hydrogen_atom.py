"""EXX hydrogen atom.

Compare self-consistent EXX calculation with non self-consistent
EXX calculation on top of LDA.
"""
from ase import Atoms
from ase.units import Ry
from gpaw import GPAW, PW
from gpaw.hybrids.eigenvalues import non_self_consistent_eigenvalues
from gpaw.hybrids.energy import non_self_consistent_energy

atoms = Atoms('H', magmoms=[1.0])
atoms.center(vacuum=5.0)

# Self-consistent calculation:
atoms.calc = GPAW(mode=PW(600),
                  xc={'name': 'EXX', 'backend': 'pw'})
                  #legacy_gpaw=not True)
eexx = atoms.get_potential_energy() + atoms.calc.get_reference_energy()

# Check energy
eexxref = -1.0 * Ry
assert abs(eexx - eexxref) < 0.001

# ... and eigenvalues
eig1, eig2 = (atoms.calc.get_eigenvalues(spin=spin)[0] for spin in [0, 1])
eigref1 = -1.0 * Ry
eigref2 = ...  # ?
assert abs(eig1 - eigref1) < 0.03
# assert abs(eig2 - eigref2) < 0.03

# LDA:
atoms.calc = GPAW(mode=PW(600),
                  xc='LDA')
atoms.get_potential_energy()

# Check non self-consistent eigenvalues
result = non_self_consistent_eigenvalues(atoms.calc,
                                         'EXX',
                                         snapshot='h-hse-snapshot.json')
eiglda, vlda, vexx = result
eig1b, eig2b = (eiglda - vlda + vexx)[:, 0, 0]
assert abs(eig1b - eig1) < 0.04
assert abs(eig2b - eig2) < 1.1

# ... and energy
energies = non_self_consistent_energy(atoms.calc, 'EXX')
eexxb = energies.sum() + atoms.calc.get_reference_energy()
assert abs(eexxb - eexx) < 0.03
