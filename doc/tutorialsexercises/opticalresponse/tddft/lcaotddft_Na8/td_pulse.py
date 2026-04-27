import numpy as np
from ase.units import Hartree, Bohr
from gpaw.external import ConstantElectricField
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.laser import GaussianPulse
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.mpi import world

# Temporal shape of the time-dependent potential
pulse = GaussianPulse(1e-5, 10e3, 1.12, 0.3, 'sin')
# Spatial shape of the time-dependent potential
ext = ConstantElectricField(Hartree / Bohr, [1., 0., 0.])
# Full time-dependent potential
td_potential = {'ext': ext, 'laser': pulse}

# Write the temporal shape to a file
pulse.write('pulse.dat', np.arange(0, 30e3, 10.0), world=world)

# Set up the time-propagation calculation
td_calc = LCAOTDDFT('gs.gpw', td_potential=td_potential,
                    txt='tdpulse.out')

# Attach the data recording and analysis tools
DipoleMomentWriter(td_calc, 'dmpulse.dat')

# Propagate
td_calc.propagate(20, 1500)

# Save the state for restarting later
td_calc.write('tdpulse.gpw', mode='all')
