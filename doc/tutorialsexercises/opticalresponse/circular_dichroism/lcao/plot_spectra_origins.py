# web-page: spectra_origins.png
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 6 / 1.62))
for ls, gauge in [('r', 'length'), ('b', 'velocity')]:
    for tag in ['COM', 'COM+x', 'COM+y', 'COM+z', '123']:
        data_ej = np.loadtxt(f'rot_spec-{tag}_{gauge}.dat')
        ax.plot(data_ej[:, 0], data_ej[:, 1], ls, label=gauge if tag == 'COM' else None)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.title(f'Rotatory strength of (R)-methyloxirane')
plt.xlabel('Energy (eV)')
plt.legend(loc='upper left', title='Different origins')
plt.ylabel(r'R (10$^{-40}$ cgs eV$^{-1})$')
plt.xlim(5, 10)
plt.ylim(-90, 90)
plt.tight_layout()
plt.savefig(f'spectra_origins.png')
