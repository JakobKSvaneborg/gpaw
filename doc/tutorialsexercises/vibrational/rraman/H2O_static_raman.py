from ase.vibrations.raman import StaticRamanCalculator
from ase.io import read

from gpaw import GPAW, FermiDirac
from gpaw.utilities.adjust_cell import adjust_cell
from gpaw.external import static_polarizability


h = 0.2
xc = 'PBE'
atoms = read('relaxed.traj')
adjust_cell(atoms, 4., h=h)

atoms.calc = GPAW(mode='fd', xc=xc, h=h,
                  occupations=FermiDirac(width=0.1),
                  symmetry={'point_group': False})
atoms.get_potential_energy()


class Polarizability:
    def __call__(self, atoms):
        return static_polarizability(atoms)


name = exname = 'static_raman'
ram = StaticRamanCalculator(
    atoms, Polarizability, name=name)
ram.run()

ram.summary()
