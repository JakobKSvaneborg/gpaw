from gpaw import GPAW
from pathlib import Path
from gpaw.borncharges import born_charges_wf

gpw_file = Path('BaTiO3.gpw')
calc = GPAW(gpw_file, txt=None)
atoms = calc.get_atoms()

atoms.calc = calc
born_charges_wf(atoms, gpw_file=gpw_file)
