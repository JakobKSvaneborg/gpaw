import numpy as np
import matplotlib.pyplot as plt

# Load the saved GWQEH results
data = np.load('MoS2_gwqeh_out_qeh.npz', allow_pickle=True)

# Extract quasiparticle corrections and self-energies
sigma_sin = data['sigma_sin']
qp_sin = data['qp_sin']
bands = data['bands']
b1, b2 = bands
band_indices = np.arange(b1, b2)
nval = 10 - b1  # MoS2 has 18 electrons -> 9 occupied bands, band 9 is VBM

# Print corrections
print('G Delta-W QEH Quasiparticle Corrections for bilayer MoS2')
print('=' * 55)
for i, band in enumerate(band_indices):
    label = 'VB' if band < 10 else 'CB'
    print(f'  Band {band} ({label}): Delta Sigma = {sigma_sin[0, 0, i]:.4f} Ha'
          f',  Delta E_QP = {qp_sin[0, 0, i]:.4f} eV')

# Compute band gap reduction
vb_max_correction = qp_sin[0, 0, :nval].max()
cb_min_correction = qp_sin[0, 0, nval:].min()
gap_reduction = vb_max_correction - cb_min_correction
print(f'\nBand gap reduction: {gap_reduction:.3f} eV')

# Plot
fig, ax = plt.subplots(figsize=(6, 4))
colors = ['#2196F3' if b < 10 else '#F44336' for b in band_indices]
ax.bar(band_indices, qp_sin[0, 0, :], color=colors, edgecolor='black',
       linewidth=0.5)
ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax.set_xlabel('Band index')
ax.set_ylabel(r'$\Delta E^{\mathrm{QP}}$ (eV)')
ax.set_title(r'G$\Delta$W-QEH correction: MoS$_2$ in bilayer MoS$_2$')
ax.set_xticks(band_indices)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2196F3', edgecolor='black',
                         label='Valence'),
                   Patch(facecolor='#F44336', edgecolor='black',
                         label='Conduction')]
ax.legend(handles=legend_elements)
plt.tight_layout()
plt.savefig('MoS2_qp_corrections.png', dpi=150)
print('\nPlot saved to MoS2_qp_corrections.png')
