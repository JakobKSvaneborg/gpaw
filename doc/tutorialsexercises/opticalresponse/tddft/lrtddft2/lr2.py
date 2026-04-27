# web-page: lr2_with_07.00eV.out
# web-page: spectrum_with_07.00eV.dat
# web-page: transitions_with_07.00eV.dat
# Start
from gpaw.mpi import world
from gpaw import GPAW
from gpaw.lrtddft2 import LrTDDFT2
from gpaw.lrtddft2.lr_communicators import LrCommunicators

# Maximum energy difference for Kohn-Sham transitions
# included in the calculation
max_energy_diff = 7.0  # eV

# Set parallelization
dd_size = 2 * 2 * 2
eh_size = world.size // dd_size
assert eh_size * dd_size == world.size
lr_comms = LrCommunicators(world, dd_size, eh_size)

calc = GPAW('unocc.gpw',
            communicator=lr_comms.dd_comm,
            legacy_gpaw=True)
lr = LrTDDFT2('lr2', calc,
              fxc='LDA',
              max_energy_diff=max_energy_diff,
              recalculate=None,  # Change this to force recalculation
              lr_communicators=lr_comms,
              txt='lr2_with_%05.2feV.out' % max_energy_diff)

# This is the expensive part triggering the calculation!
lr.calculate()

# Get and write spectrum
spec = lr.get_spectrum('spectrum_with_%05.2feV.dat'
                       % max_energy_diff,
                       min_energy=0,
                       max_energy=10,
                       energy_step=0.01,
                       width=0.1)

# Get and write transitions
trans = lr.get_transitions('transitions_with_%05.2feV.dat'
                           % max_energy_diff,
                           min_energy=0.0,
                           max_energy=10.0)
