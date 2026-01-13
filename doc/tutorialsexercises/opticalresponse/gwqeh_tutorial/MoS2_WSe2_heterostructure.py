from gpaw.response.gwqeh import GWQEHCorrection

# Define MoS2/WSe2 bilayer heterostructure
# Building blocks for each layer (these need to be pre-calculated)
structure = ['MoS2-chi.npz', 'WSe2-chi.npz']

# Interlayer distance (typical for MoS2/WSe2)
d = [6.5]  # Angstrom

# Calculate QP correction for MoS2 layer (layer=0)
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

# To calculate for WSe2 layer, you would use:
# gwq_wse2 = GWQEHCorrection(calc='WSe2_fulldiag.gpw',
#                            filename='WSe2_in_hetero',
#                            kpts=[0],
#                            bands=(8, 12),
#                            structure=structure,
#                            d=d,
#                            layer=1,  # WSe2 is layer 1
#                            domega0=0.025,
#                            omega2=10.0)
# qp_wse2 = gwq_wse2.calculate_qp_correction()
