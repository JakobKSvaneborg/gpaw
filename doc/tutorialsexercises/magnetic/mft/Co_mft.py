"""Compute the isotropic exchange constants of Co on a band path between
selected high-symmetry points of the hcp lattice."""

# General modules
import numpy as np

# Script modules
from ase.data import covalent_radii

from gpaw import restart
from gpaw.mpi import world
from gpaw.response import ResponseContext, ResponseGroundStateAdapter
from gpaw.response.site_kernels import (SphericalSiteKernels,
                                        CylindricalSiteKernels,
                                        ParallelepipedicSiteKernels)
from gpaw.response.chiks import ChiKSCalculator
from gpaw.response.localft import LocalPAWFTCalculator
from gpaw.response.mft import IsotropicExchangeCalculator


# ---------- Inputs ---------- #

# Parameters reused from the ground state
gpw = 'Co.gpw'
kpts = 24
nbands = 2 * 6

# Convergence criteria of Kohn-Sham orbitals
conv = {'bands': nbands,
        'eigenstates': 1.e-14}

# We choose the plane wave energy cutoff of the mft response calculation
# so as to provide magnon energies converged within 5%, based on the
# convergence study of Co(fcc) in [arXiv:2204.04169]
ecut = 750  # eV

# We map out the high-symmetry path G-M-K-G-A, by generating all commensurate
# q-vectors on the path
qGM_qc = np.array([[x / kpts, 0., 0.]
                   for x in range(kpts // 2 + 1)])
qMK_qc = np.array([[1 / 2. - x / kpts, 2 * x / kpts, 0.]
                   for x in range(kpts // 6 + 1)])
qKG_qc = np.array([[x / kpts, x / kpts, 0.]
                   for x in reversed(range(kpts // 3 + 1))])
qGA_qc = np.array([[0., 0., x / kpts]
                   for x in range(kpts // 2 + 1)])
q_qc = np.vstack([qGM_qc, qMK_qc[1:], qKG_qc[1:], qGA_qc[1:]])

# We define several Co site radii to try for the spherical site kernels
rc_r = np.linspace(0.5, 1.75, 51)

# ---------- Script ---------- #

# Recalculate the Kohn-Sham orbitals
atoms, calc = restart(gpw, parallel={'domain': 1})
calc = calc.fixed_density(convergence=conv, txt='Co_es.txt')

# Initialize the context and gs adapter
context = ResponseContext(txt='Co_mft.txt')
gs = ResponseGroundStateAdapter(calc)

# Initialize the ChiKSCalculator
chiks_calc = ChiKSCalculator(gs, context,
                             ecut=ecut, nbands=nbands,
                             gammacentered=True)

# Initialize the localft calculator
localft_calc = LocalPAWFTCalculator(gs, context)

# Initialize the exchange calculator
isoexch_calc = IsotropicExchangeCalculator(chiks_calc, localft_calc)

# Initialize site kernels with two sublattices
positions = atoms.positions  # sublattice positions
# Create two Co sublattices with spherical sites, but vary only one radius
rc1_pa = np.array([[rc, 1.2] for rc in rc_r])
sph_sitekernels1 = SphericalSiteKernels(positions, rc1_pa)
assert sph_sitekernels1.nsites == 2  # Check to illustrate SiteKernels magic
assert sph_sitekernels1.npartitions == len(rc_r)
# Create two Co sublattices with spherical sites of equal size
rc2_pa = np.array([[rc, rc] for rc in rc_r])
# We could initialize the site kernel instance normally as follows
sph_sitekernels2 = SphericalSiteKernels(positions, rc2_pa)
# However, there is also a second option. We can initialize a site kernels
# instance for each sublattice and then add them together
sph_sitekernels2 = SphericalSiteKernels(positions[:1], rc2_pa[:, :1])\
    + SphericalSiteKernels(positions[1:], rc2_pa[:, 1:])
assert sph_sitekernels2.nsites == 2
assert sph_sitekernels2.npartitions == len(rc_r)

# Initialize site kernels with a single sublattice
# Create a single site spanning the entire unit cell
cell_cv = atoms.get_cell()
center_v = np.sum(cell_cv, axis=0) / 2.
uc_sitekernels = ParallelepipedicSiteKernels([center_v], [[cell_cv]])
# Create a single site as a cylinder encapsulating both Co atoms
d_v = positions[1] - positions[0]  # displacement vector between atoms
d = np.linalg.norm(d_v)
ez_v = d_v / d  # normalized cylinder axis
# Use the average covalent radii as the cylinder radius
rc = np.average([covalent_radii[n] for n in atoms.get_atomic_numbers()])
# Cylinder height
hc = d + 2 * rc
cyl_sitekernels = CylindricalSiteKernels([center_v], [[ez_v]], [[rc]], [[hc]])
# Because both the unit cell and cylindrical site kernels are placed at the
# same position (the cell center), we may view them as spatial partitionings
# of the same Heisenberg model. Therefore, we can append one site kernel to
# the other, which will create a new "partition".
mix_sitekernels = uc_sitekernels.copy()  # Make a fresh copy
mix_sitekernels.append(cyl_sitekernels)
assert mix_sitekernels.nsites == 1
assert mix_sitekernels.npartitions == 2

# Allocate arrays for the exchange constants
nq = len(q_qc)
Jsph1_qabr = np.empty((nq, 2, 2, len(rc_r)), dtype=complex)
Jsph2_qabr = np.empty((nq, 2, 2, len(rc_r)), dtype=complex)
Jmix_qp = np.empty((nq, 2), dtype=complex)

# Compute the isotropic exchange coupling along the chosen bandpath
for q, q_c in enumerate(q_qc):
    Jsph1_qabr[q] = isoexch_calc(q_c, sph_sitekernels1)
    Jsph2_qabr[q] = isoexch_calc(q_c, sph_sitekernels2)
    Jmix_qp[q] = isoexch_calc(q_c, mix_sitekernels)[0, 0, :]  # nsites == 1

# Save the bandpath, spherical radii and computed exchange constants
if world.rank == 0:
    np.save('Co_q_qc.npy', q_qc)
    np.save('Co_rc_r.npy', rc_r)
    np.save('Co_Jsph1_qabr.npy', Jsph1_qabr)
    np.save('Co_Jsph2_qabr.npy', Jsph2_qabr)
    np.save('Co_Jmix_qp.npy', Jmix_qp)
