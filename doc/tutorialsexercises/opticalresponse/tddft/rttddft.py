# P1
# Sodium dimer
from ase.build import molecule
atoms = molecule('Na2')
atoms.center(vacuum=8.0)

# Ground-state calculation
from gpaw import GPAW
calc = GPAW(mode='lcao', h=0.3, basis='dzp',
            setups={'Na': '1'},
            convergence={'density': 1e-12},
            symmetry={'point_group': False})
atoms.calc = calc
energy = atoms.get_potential_energy()
calc.write('gs.gpw', mode='all')
# P2
# Time-propagation calculation
from ase.parallel import paropen
from gpaw.new.rttddft import RTTDDFT

# Read converged ground-state file
td_calc = RTTDDFT.from_file('gs.gpw', td_algorithm='sicn')

# Kick
td_calc.absorption_kick([0.0, 0.0, 1e-5])

dt = 1e-3  # Time step in ASE units Å sqrt(u/eV). Rougly equal to 10as
# Open dipole moment file for writing and propagate
with paropen('dm.dat', 'w') as fp:
    fp.write('# %15s %15s %22s %22s %22s\n' %
             ('time', 'norm', 'dmx', 'dmy', 'dmz'))
    fp.write('# Kick = [%22.12le, %22.12le, %22.12le]; \n'
             % tuple(td_calc.history.kicks[0].strength))
    for result in td_calc.ipropagate(dt, 3000):
        fp.write('%20.8lf %20.8le %22.12le %22.12le %22.12le\n' %
                 (result.time, 0, *result.dipolemoment))

# Save the state for restarting later
td_calc.write('td.gpw')
# P3
# Analyze the results
from gpaw.tddft.spectrum import photoabsorption_spectrum
photoabsorption_spectrum('dm.dat', 'spec.dat')
