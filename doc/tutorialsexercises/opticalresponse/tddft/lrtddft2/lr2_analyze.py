# web-page: tc_000_with_08.00eV.txt
# Start
from ase.parallel import paropen
from gpaw import GPAW
from gpaw.lrtddft2 import LrTDDFT2

# Maximum energy difference for Kohn-Sham transitions
# included in the calculation
max_energy_diff = 8.0  # eV

calc = GPAW('unocc.gpw', legacy_gpaw=True)
lr = LrTDDFT2('lr2', calc,
              fxc='LDA',
              max_energy_diff=max_energy_diff,
              txt='-')

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

# Get and write transition contributions
index = 0
f2 = lr.get_transition_contributions(index_of_transition=index)
with paropen('tc_%03d_with_%05.2feV.txt'
             % (index, max_energy_diff), 'w') as f:
    f.write('Transition %d at %.2f eV\n' % (index, trans[0][index]))
    f.write(' %5s => %5s  contribution\n' % ('occ', 'unocc'))
    for (ip, val) in enumerate(f2):
        if (val > 1e-3):
            f.write(' %5d => %5d  %8.4f%%\n' %
                    (lr.ks_singles.kss_list[ip].occ_ind,
                     lr.ks_singles.kss_list[ip].unocc_ind,
                     val / 2. * 100))
