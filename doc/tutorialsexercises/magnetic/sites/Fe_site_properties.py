"""Calculate the site magnetization and Zeeman energy, based on the ground
state of Fe(bcc)."""

import numpy as np

from ase.build import bulk
from gpaw import GPAW, PW, FermiDirac
from gpaw.mpi import rank
from gpaw.response import ResponseGroundStateAdapter
from gpaw.response.site_data import AtomicSites, AtomicSiteData
from gpaw.response.mft import (calculate_site_magnetization,
                               calculate_site_zeeman_energy)

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
rmin_a, rmax_a = AtomicSiteData.valid_site_radii_range(gs)
# We can then define a range of site configurations to investigate
rc_r = np.linspace(rmin_a[0], rmax_a[0], 51)
sites = AtomicSites(
    indices=[0],  # indices of the magnetic atoms
    radii=[rc_r],  # spherical cutoff radii for each magnetic atom
)
# and calculate the site properties of interest
m_ar = calculate_site_magnetization(gs, sites)
EZ_ar = calculate_site_zeeman_energy(gs, sites)

# Save site data
if rank == 0:
    np.save('rc_r.npy', rc_r)
    np.save('m_r.npy', m_ar[0])
    np.save('EZ_r.npy', EZ_ar[0])
