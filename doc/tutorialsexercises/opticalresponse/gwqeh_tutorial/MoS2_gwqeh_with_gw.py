"""Combine monolayer GW with G Delta-W QEH heterostructure correction.

This gives the most accurate quasiparticle energies: the monolayer GW
calculation provides the self-energy and renormalization factor Z for
the isolated layer, while the G Delta-W QEH correction accounts for
the additional screening from neighboring layers.

The final QP energies are:
    E_QP^HS = E_QP^mono + Z * Delta Sigma
"""
from gpaw.response.g0w0 import G0W0
from gpaw.response.gwqeh import GWQEHCorrection

# Step A: Run standard G0W0 calculation for isolated MoS2 monolayer.
# This provides the monolayer QP energies and the Z factor.
gw = G0W0(calc='MoS2_fulldiag.gpw',
          bands=(8, 12),
          ecut=80,
          truncation='2D',
          nblocksmax=True,
          q0_correction=True,
          filename='MoS2_gw')

gw.calculate()

# Step B: Run G Delta-W QEH calculation using GW results.
# The gwfile provides Z from the monolayer GW calculation.
structure = ['MoS2-chi.npz', 'MoS2-chi.npz']
d = [6.15]

gwq = GWQEHCorrection(calc='MoS2_fulldiag.gpw',
                      gwfile='MoS2_gw_results_GW.pckl',
                      filename='MoS2_gwqeh_full',
                      kpts=[0],
                      bands=(8, 12),
                      structure=structure,
                      d=d,
                      layer=0,
                      domega0=0.025,
                      omega2=10.0)

# Get full QP energies: E_QP^mono + Delta E_QP
qp_energies = gwq.calculate_qp_energies()

print('Full quasiparticle energies (monolayer GW + heterostructure correction):')
for i, band in enumerate(range(8, 12)):
    print(f'  Band {band}: {qp_energies[0, 0, i]:.4f} eV')
