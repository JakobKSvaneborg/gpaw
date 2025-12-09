import numpy as np
from gpaw import GPAW
from gpaw.mpi import world
from gpaw.response import ResponseGroundStateAdapter, ResponseContext
from gpaw.response.mft import HeisenbergExchangeCalculator
from gpaw.response.site_data import AtomicSites, get_site_radii_range

gpw = 'gs_afm.gpw'
nbands = 60

calc = GPAW(gpw)
q_qc = calc.get_bz_k_points()

context = ResponseContext(txt='mft.txt')
gs = ResponseGroundStateAdapter.from_gpw_file(gpw)
atoms = gs.atoms

# Initialize the spherical sites
_, r_a = get_site_radii_range(gs)
r0 = np.min([r_a[0], r_a[1]])
sites = AtomicSites(indices=[0, 1], radii=[[r0], [r0]])

# Compute the isotropic exchange coupling along the chosen bandpath
jcalc = HeisenbergExchangeCalculator(
    gs, sites, context=context, nbands=nbands)
J_qab = np.array([jcalc(q_c).array[..., 0] for q_c in q_qc])
context.write_timer()

# Save the bandpath, spherical radii and computed exchange constants
if world.rank == 0:
    np.save('q_qc.npy', q_qc)
    np.save('J_n%s_qab.npy' % nbands, J_qab)
