from ase.io import read
from gpaw import setup_paths
from gpaw import GPAW

# Insert the path to the created basis set
setup_paths.insert(0, '.')

atoms = read('r-methyloxirane.xyz')
atoms.center(vacuum=6.0)

calc = GPAW(mode='lcao',
            basis='aug.dzp',
            h=0.3,
            xc='PBE',
            nbands=16,
            poissonsolver={
                'name': 'MomentCorrectionPoissonSolver',
                'poissonsolver': 'fast',
                'moment_corrections': 1 + 3 + 5},
            convergence={'density': 1e-12},
            txt='gs.out',
            symmetry={'point_group': False},
            legacy_gpaw=True)
atoms.calc = calc
atoms.get_potential_energy()
calc.write('gs.gpw', mode='all')
