# P1
# Sodium dimer
from ase.build import molecule
atoms = molecule('Na2')
atoms.center(vacuum=8.0)

# Poisson solver with multipole corrections up to l=2
poissonsolver = {'name': 'MomentCorrectionPoissonSolver',
                 'moment_corrections': 1 + 3 + 5,
                 'poissonsolver': 'fast'}

# Ground-state calculation
from gpaw.new.ase_interface import GPAW
calc = GPAW(mode='lcao', h=0.3, basis='dzp',
            setups={'Na': '1'},
            poissonsolver=poissonsolver,
            convergence={'density': 1e-12},
            symmetry={'point_group': False})
atoms.calc = calc
energy = atoms.get_potential_energy()
calc.write('gs.gpw', mode='all')
# P2
# Time-propagation calculation
from gpaw.external import create_absorption_kick
from gpaw.new.rttddft import RTTDDFT
from gpaw.new.rttddft.writers import DipoleMomentWriter

dt = 1e-3  # Time step in ASE units Å sqrt(u/eV). Rougly equal to 10as

# Read converged ground-state file
td_calc = RTTDDFT.from_file('gs.gpw', td_algorithm='sicn')

# Open dipole moment file
with DipoleMomentWriter('dm.dat') as dmwriter:
    # Optionally, write a comment to signify (re)start
    dmwriter.write_comment(f'Start at {td_calc.time:.8f}')

    # Kick, write a comment about the kick, and write the dipole moment
    kick_potential = create_absorption_kick([0.0, 0.0, 1e-5])
    td_calc.kick(kick_potential)
    dmwriter.write_kick(td_calc.time, kick_potential)
    dmwriter.write_dm(td_calc.time, td_calc.state, td_calc.pot_calc)

    # Propagate for 3000 steps
    for _ in td_calc.ipropagate(dt, 3000):
        dmwriter.write_dm(td_calc.time, td_calc.state, td_calc.pot_calc)

# Save the state for restarting later
td_calc.write('td.gpw')
# P3
# Analyze the results
from gpaw.tddft.spectrum import photoabsorption_spectrum
photoabsorption_spectrum('dm.dat', 'spec.dat')
