from ase import Atoms
from gpaw import GPAW
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.qed import RRemission
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.tddft.spectrum import photoabsorption_spectrum

# Sodium dimer chain
d = 1.104  # N2 bondlength
L = 8.0  # N2-N2 distance
atoms = Atoms('Na4',
              positions=([[L, 0, +d / 2], [L, 0, -d / 2], [2 * L, 0, +d / 2],
                         [2 * L, 0, -d / 2]]), cell=[L + 2 * L, 8.0, 8.0])
atoms.center()

# Ground-state calculation
calc = GPAW(mode='lcao', h=0.3, basis='dzp', xc='LDA',
            setups={'Na': '1'},
            convergence={'density': 1e-12},
            txt='gs_nad2.out',
            symmetry={'point_group': False})
atoms.calc = calc
energy = atoms.get_potential_energy()
calc.write('gs.gpw', mode='all')

# Time-propagation calculation
# Read converged ground-state file
env = {'environment': 'waveguide',
       'quantization_plane': 35.05,
       'cavity_polarization': [0, 0, 1]}
td_calc = LCAOTDDFT('gs.gpw', rremission=RRemission(env))
# Attach any data recording or analysis tools
DipoleMomentWriter(td_calc, 'dm_nad2.dat')
# Kick
td_calc.absorption_kick([0.0, 0.0, 1e-5])
# Propagate
td_calc.propagate(20, 2500)
# Please note that those parameter values are quite course
# and should be properly converged for subsequent applications.

# Calculate spectrum with small artificial broadening
photoabsorption_spectrum('dm_nad2.dat', 'spec_nad2.dat',
                         folding='Gauss', width=0.005,
                         e_min=0.0, e_max=10.0, delta_e=0.005)
