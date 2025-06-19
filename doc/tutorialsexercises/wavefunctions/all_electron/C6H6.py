from gpaw import GPAW
from ase.build import molecule

calc = GPAW(mode='fd')
mol = molecule('C6H6', calculator=calc)
mol.center(vacuum=5)
E = mol.get_potential_energy()
nt = calc.get_pseudo_density()
n_ae = calc.get_all_electron_density()
# --- literalinclude division line ---
n_ae_fine = calc.get_all_electron_density(gridrefinement=2)
