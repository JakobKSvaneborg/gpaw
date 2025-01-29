from ase.build import mx2

from gpaw import GPAW, FermiDirac

name = 'HfS2'
atoms = mx2(formula=name, kind='1T', a=3.648, thickness=2.895,
            size=(1, 1, 1), vacuum=12.0)
atoms.center(vacuum=6.0, axis=2)

h = 0.18
kx = 9
ky = 9
kz = 1

calc = GPAW(mode='lcao',
            h=h,
            kpts={'size': (kx, ky, kz), 'gamma': True},
            xc='PBE',
            basis='dzp',
            parallel={'band': 1},
            symmetry='off',
            convergence={'bands': -2},
            maxiter=600,
            txt=None,
            occupations=FermiDirac(width=0.01))

atoms.calc = calc
atoms.get_potential_energy()
calc.write(name + '.gpw')
