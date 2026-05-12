"""Example: G Delta-W correction for MoS2 in a MoS2/WSe2 heterostructure.

This script demonstrates how to compute band structure corrections for
a layer embedded in a heterostructure made of different materials. Each
material needs its own groundstate calculation and building block file.

Note: This example requires pre-calculated building blocks for both
MoS2 and WSe2. The WSe2 building block can be generated following the
same procedure as for MoS2 (Steps 1-2 in the tutorial).
"""
from gpaw.response.gwqeh import GWQEHCorrection

# Define a MoS2/WSe2 bilayer heterostructure.
# The structure list contains the building block file for each layer,
# ordered from bottom to top.
structure = ['MoS2-chi.npz', 'WSe2-chi.npz']

# Interlayer distance between MoS2 and WSe2 (Angstrom).
# This is the distance between the metal atom planes of the two layers.
d = [6.5]

# Calculate the QP correction for the MoS2 layer (layer index 0).
# The 'calc' argument points to the MoS2 groundstate.
gwq_mos2 = GWQEHCorrection(calc='MoS2_fulldiag.gpw',
                           filename='MoS2_in_hetero',
                           kpts=[0],
                           bands=(8, 12),
                           structure=structure,
                           d=d,
                           layer=0,
                           domega0=0.025,
                           omega2=10.0)

qp_mos2 = gwq_mos2.calculate_qp_correction()

print('MoS2 layer in MoS2/WSe2 heterostructure:')
for i, band in enumerate(range(8, 12)):
    print(f'  Band {band}: {qp_mos2[0, 0, i]:.4f} eV')

# To compute the correction for the WSe2 layer, use layer=1
# and point 'calc' to the WSe2 groundstate:
#
# gwq_wse2 = GWQEHCorrection(calc='WSe2_fulldiag.gpw',
#                            filename='WSe2_in_hetero',
#                            kpts=[0],
#                            bands=(8, 12),
#                            structure=structure,
#                            d=d,
#                            layer=1,
#                            domega0=0.025,
#                            omega2=10.0)
# qp_wse2 = gwq_wse2.calculate_qp_correction()
