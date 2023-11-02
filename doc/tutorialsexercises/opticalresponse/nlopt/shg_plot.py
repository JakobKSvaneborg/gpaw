# web-page: shg.png
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read

# Plot and save both spectra
atoms = read('gs.txt')
cell = atoms.get_cell()
cellsize = atoms.cell.cellpar()
mult = cellsize[2] * 1e-10  # Make the sheet susceptibility
legls = []
res_name = ['shg_yyy_lg.npy', 'shg_yyy_vg.npy']
plt.figure(figsize=(6.0, 4.0), dpi=300)
for ii, name in enumerate(res_name):
    # Load the data
    shg = np.load(name)
    w_l = shg[0].real
    plt.plot(w_l, np.real(mult * shg[1] * 1e18), '-')
    plt.plot(w_l, np.imag(mult * shg[1] * 1e18), '--')
    legls.append(f'{name}: Re')
    legls.append(f'{name}: Im')
    plt.xlabel(r'$\hbar\omega$ (eV)')
    plt.ylabel(r'$\chi_{yyy}$ (nm$^2$/V)')
    plt.legend(legls, ncol=2)
plt.tight_layout()
plt.savefig('shg.png', dpi=300)
