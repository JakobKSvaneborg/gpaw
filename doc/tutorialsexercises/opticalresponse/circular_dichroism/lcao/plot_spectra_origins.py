# web-page: spectra_origins.png
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 6 / 1.62))

for gauge in ['length', 'velocity']:
    plt.clf()
    for tag in ['COM', 'COM+x', 'COM+y', 'COM+z', '123']:
        data_ej = np.loadtxt(f'rot_spec-{tag}_{gauge}.dat')
        ax.plot(data_ej[:, 0], data_ej[:, 1], label=tag)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.title(f'Rotatory strength of (R)-methyloxirane, {gauge}')
    plt.xlabel('Energy (eV)')
    plt.legend(loc='upper left', title='Origin')
    plt.ylabel(r'R (10$^{-40}$ cgs eV$^{-1})$')
    plt.xlim(0, 10)
    plt.ylim(-80, 80)
    plt.tight_layout()
    plt.savefig(f'spectra_origins_{gauge}.png')
