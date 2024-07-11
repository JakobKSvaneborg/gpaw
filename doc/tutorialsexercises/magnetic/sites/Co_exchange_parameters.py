"""Calculate the site magnetization and exchange parameters of Co (hcp)."""

import numpy as np

from ase.build import bulk
from gpaw import GPAW, PW, FermiDirac
from gpaw.response import ResponseGroundStateAdapter, ResponseContext
from gpaw.response.site_data import (AtomicSites, get_site_radii_range,
                                     maximize_site_magnetization)
from gpaw.response.mft import calculate_exchange_parameters

# ----- Ground state calculation ----- #

# Set up crystal structure
a = 2.5071
c = 4.0695
mm = 1.67
atoms = bulk('Co', 'hcp', a=a, c=c)
atoms.set_initial_magnetic_moments([mm, mm])
atoms.center()

# Perform ground state calculation
kpts = 24
nbands = 2 * 6  # natoms * (4s + 3d)
calc = GPAW(xc='LDA',
            mode=PW(800),
            kpts={'size': (kpts, kpts, kpts), 'gamma': True},
            nbands=nbands + 8,  # add some extra bands for smooth converence
            # We converge the ground state density tightly
            convergence={'density': 1e-8,
                         'eigenstates': 1e-14,
                         'bands': nbands},
            occupations=FermiDirac(0.001),
            parallel={'domain': 1},
            txt='Co_gs.txt')
atoms.calc = calc
atoms.get_potential_energy()
gs = ResponseGroundStateAdapter(calc)

# ----- Site properties as a function of rc ----- #

# Find ideal cutoff radii
rc_a, magmom_a = maximize_site_magnetization(gs)
# which should be the same for the two Co atoms due to symmetry
assert abs(rc_a[1] - rc_a[0]) < 1e-6
assert abs(magmom_a[1] - magmom_a[0]) < 1e-6

# Get the valid site radii range
rmin_a, rmax_a = get_site_radii_range(gs)
# which again should be identical for the two Co atoms
assert abs(rmin_a[1] - rmin_a[0]) < 1e-6
assert abs(rmax_a[1] - rmax_a[0]) < 1e-6
rc_r = np.linspace(rmin_a[0], rmax_a[0], 51)  # partitionings
rc_ar = [rc_r] * 2

# Set up magnetic sites and calculate the exchange parameters at the
# high-symmetry points
context = ResponseContext('Co_hsp.txt')
sites = AtomicSites(indices=[0, 1], radii=rc_ar)
q_pc = np.array([[0., 0., 0],  # Γ
                 [0.5, 0., 0.],  # M
                 [1 / 3., 1 / 3., 0.],  # K
                 [0., 0., 0.5]])  # A
J_pabr = np.array(
    [calculate_exchange_parameters(
        gs, sites, q_c, context=context, nbands=nbands)
     for q_c in q_pc])

# ----- Save results ----- #

if context.comm.rank == 0:
    np.save('rc.npy', rc_a[0])
    np.save('magmom.npy', magmom_a[0])
    np.save('rc_r.npy', rc_r)
    np.save('q_pc.npy', q_pc)
    np.save('J_pabr.npy', J_pabr)
