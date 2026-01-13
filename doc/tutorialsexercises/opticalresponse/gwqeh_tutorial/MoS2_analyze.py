import numpy as np

# Load the saved GWQEH results
data = np.load('MoS2_gwqeh_out_qeh.npz')

# Extract quasiparticle corrections
qp_sin = data['qp_sin']
bands = data['bands']

print('GWQEH Quasiparticle Corrections')
print('=' * 40)

# Print corrections for each band
b1, b2 = bands
for i, band in enumerate(range(b1, b2)):
    qp = qp_sin[0, 0, i]  # spin=0, kpt=0
    print(f'Band index: {band}, QP correction: {qp:.2f} eV')

# Check if calculation completed
if data['complete']:
    print('\nCalculation completed successfully.')
else:
    print('\nWarning: Calculation may not be complete.')
