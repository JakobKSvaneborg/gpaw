import numpy as np
from gpaw.mpi import rank
from gpaw.response import ResponseGroundStateAdapter, ResponseContext
from gpaw.response.mft import HeisenbergExchangeCalculator
from gpaw.response.site_data import AtomicSites, get_site_radii_range

gpw = 'gs_afm.gpw'
nbands = 60

kpts = 4
qXG_qc = np.array([[1 / 2 - x / kpts, 0, 0]
                   for x in range(kpts // 2 + 1)])
qGM_qc = np.array([[x / kpts, x / kpts, 0]
                   for x in range(kpts // 2 + 1)])
qMA_qc = np.array([[1 / 2, 1 / 2, x / kpts]
                   for x in range(kpts // 2 + 1)])
qAZ_qc = np.array([[1 / 2 - x / kpts, 1 / 2 - x / kpts, 1 / 2]
                   for x in range(kpts // 2 + 1)])
qZR_qc = np.array([[x / kpts, 0, 1 / 2]
                   for x in range(kpts // 2 + 1)])
q_qc = np.vstack([qXG_qc, qGM_qc[1:], qMA_qc[1:], qAZ_qc[1:], qZR_qc[1:]])
print(len(q_qc))

context = ResponseContext(txt='mft_q.txt')
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
if rank == 0:
    np.save('q_qc_path.npy', q_qc)
    np.save('J_60_qab_path.npy', J_qab)
