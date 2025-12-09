"""Calculate the site magnetization and Zeeman energy, based on the ground
state of Fe(bcc)."""

import numpy as np

from ase.build import bulk
from gpaw import GPAW, PW, FermiDirac
from gpaw.mpi import world
from gpaw.response import ResponseGroundStateAdapter
from gpaw.response.site_data import (AtomicSites, get_site_radii_range,
                                     calculate_site_magnetization,
                                     calculate_site_zeeman_energy,
                                     maximize_site_magnetization)

# ----- Ground state calculation ----- #

# Set up crystal structure
a = 2.867  # Lattice constant
mm = 2.21  # Initial magnetic moment
atoms = bulk('Fe', 'bcc', a=a)
atoms.set_initial_magnetic_moments([mm])
atoms.center()

# Perform ground state calculation
calc = GPAW(xc='LDA',
            mode=PW(800),
            kpts={'size': (16, 16, 16), 'gamma': True},
            # We converge the ground state density tightly
            convergence={'density': 1.e-8},
            occupations=FermiDirac(0.001),
            txt='Fe_gs.txt')
atoms.calc = calc
atoms.get_potential_energy()
calc.write('Fe.gpw')

# ----- Site properties ----- #

# Due to implementational details, the choice of spherical radii is restricted
# to a certain range (to assure that each site volume can be truncated smoothly
# and does not overlap with neighbouring augmentation spheres). This range can
# be easily extracted from a given ground state:
gs = ResponseGroundStateAdapter(calc)
rmin_a, rmax_a = get_site_radii_range(gs)
# We can then define a range of site configurations to investigate
rc_r = np.linspace(rmin_a[0], rmax_a[0], 51)
sites = AtomicSites(
    indices=[0],  # indices of the magnetic atoms
    radii=[rc_r],  # spherical cutoff radii for each magnetic atom
)
# and calculate the site properties of interest
m_ar = calculate_site_magnetization(gs, sites)
EZ_ar = calculate_site_zeeman_energy(gs, sites)

# Similarly, we may also seek to identify the cutoff radius, which maximizes
# the site magnetization
rc_a, magmom_a = maximize_site_magnetization(gs)

# Save site data
if world.rank == 0:
    np.save('Fe_rc_r.npy', rc_r)
    np.save('Fe_m_r.npy', m_ar[0])
    np.save('Fe_EZ_r.npy', EZ_ar[0])
    np.save('Fe_rc.npy', rc_a[0])
    np.save('Fe_magmom.npy', magmom_a[0])
