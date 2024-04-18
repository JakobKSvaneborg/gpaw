# web-page: mag1d.png, mag2d.png
from gpaw.new.ase_interface import GPAW
import matplotlib.pyplot as plt
import numpy as np

calc = GPAW('VCl2_gs.gpw')
dens = calc.dft.densities()
grid_spacing = calc.atoms.cell[2, 2] / 200
nt = dens.pseudo_densities(grid_spacing)
n = dens.all_electron_densities(grid_spacing=grid_spacing)

i = nt.desc.size[2] // 2
x, y = n.desc.xyz()[:, :, i, :2].transpose((2, 0, 1))
uv = n.data[1:3, :, :, i]
m = (uv**2).sum(0)**0.5
u, v = uv / m
fig, ax = plt.subplots()
ct = ax.contourf(x, y, m)
cbar = fig.colorbar(ct)
cbar.ax.set_ylabel('magnetization [Å$^{-3}$]')
ax.quiver(*(a[::3, ::3] for a in [x, y, u, v]))
ax.axis('equal')
ax.set_xlabel('x [Å]')
ax.set_ylabel('y [Å]')
fig.savefig('mag2d.png')

fig, ax = plt.subplots()
x, y = n.xy(1, ..., 0, i)
x, yt = nt.xy(1, ..., 0, i)
j = len(x) // 2
L = calc.atoms.cell[0, 0]
x = np.concatenate((x[j:] - L, x[:j]))
y = np.concatenate((y[j:], y[:j]))
yt = np.concatenate((yt[j:], yt[:j]))
ax.plot(x, y, label='all-electron')
ax.plot(x, yt, label='pseudo')
ax.legend()
ax.set_xlabel('x [Å]')
ax.set_ylabel('magnetization [Å$^{-3}$]')
fig.savefig('mag1d.png')
