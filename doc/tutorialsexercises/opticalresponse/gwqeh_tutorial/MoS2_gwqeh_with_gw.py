from gpaw.response.g0w0 import G0W0
from gpaw.response.gwqeh import GWQEHCorrection

# First, run standard GW calculation for isolated MoS2
gw = G0W0(calc='MoS2_fulldiag.gpw',
          bands=(8, 12),
          ecut=80,
          truncation='2D',
          nblocksmax=True,
          q0_correction=True,
          filename='MoS2_gw')

gw.calculate()

# Now run GWQEH calculation using GW results
structure = ['MoS2-chi.npz', 'MoS2-chi.npz']
d = [6.15]

gwq = GWQEHCorrection(calc='MoS2_fulldiag.gpw',
                      gwfile='MoS2_gw_results_GW.pckl',  # GW results file
                      filename='MoS2_gwqeh_full',
                      kpts=[0],
                      bands=(8, 12),
                      structure=structure,
                      d=d,
                      layer=0,
                      domega0=0.025,
                      omega2=10.0)

# Get full QP energies (GW + QEH correction)
qp_energies = gwq.calculate_qp_energies()

print('Full quasiparticle energies (GW + QEH correction):')
for i, band in enumerate(range(8, 12)):
    print(f'  Band {band}: {qp_energies[0, 0, i]:.4f} eV')
