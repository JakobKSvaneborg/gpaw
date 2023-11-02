"""Calculate the single-particle site Zeeman energy and pair site Zeeman
energy, based on the ground state of Fe(bcc)."""

import numpy as np

from gpaw import GPAW
from gpaw.mpi import rank
from gpaw.response import ResponseContext, ResponseGroundStateAdapter
from gpaw.response.site_data import AtomicSites
from gpaw.response.mft import (calculate_single_particle_site_zeeman_energy,
                               calculate_pair_site_zeeman_energy)

# ----- Ground state calculation ----- #

# Converge additional empty-shell bands
nocc = 6  # 4s, 3d
nunocc = 32
calc = GPAW('Fe.gpw', parallel={'domain': 1})  # reuse the ground state density
calc = calc.fixed_density(nbands=nocc + nunocc + 6,
                          convergence={'bands': nocc + nunocc,
                                       'eigenstates': 1.e-14},
                          txt='Fe_es.txt')

# ----- Site sum rules ----- #

# Reuse the radius grid
rc_r = np.load('rc_r.npy')
sites = AtomicSites(indices=[0], radii=[rc_r])
# Calculate the single-particle sum rule
gs = ResponseGroundStateAdapter(calc)
context = ResponseContext('Fe_sum_rules.txt')
context.print('\n\n--- Single-particle sum rule ---\n\n')
sp_EZ_ar = calculate_single_particle_site_zeeman_energy(gs, sites, context)
# Calculate the pair Zeeman energy with a varrying number of empty-shell bands
unocc_n = 4 * np.arange(9)
EZ_nr = np.empty((len(unocc_n), len(rc_r)), dtype=complex)
for n, unocc in enumerate(unocc_n):
    context.print(f'\n\n--- Two-particle sum rule with unocc={unocc} ---\n\n')
    EZ_abr = calculate_pair_site_zeeman_energy(
        gs, sites, context,
        q_c=[0., 0., 0.],  # q-vector of the pair function
        nbands=nocc + unocc,  # number of bands to include
    )
    EZ_nr[n] = EZ_abr[0, 0]

# Save site sum rule data
if rank == 0:
    np.save('sp_EZ_r.npy', sp_EZ_ar[0])
    np.save('EZ_nr.npy', EZ_nr)
