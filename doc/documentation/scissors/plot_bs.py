import matplotlib.pyplot as plt
import numpy as np
from ase.units import Ha
from matplotlib.collections import LineCollection

from gpaw import GPAW


def line_segments(x_k, y_nk):
    N, K = y_nk.shape
    S_nksv = np.empty((N, K, 3, 2))
    S_nksv[:, 0, 0, 0] = 0.0
    S_nksv[:, 1:, 0, 0] = 0.5 * (x_k[:-1] + x_k[1:])
    S_nksv[:, :, 1, 0] = x_k
    S_nksv[:, :-1, 2, 0] = 0.5 * (x_k[:-1] + x_k[1:])
    S_nksv[:, -1, 2, 0] = x_k[-1]
    S_nksv[:, 0, 0, 1] = np.nan
    S_nksv[:, 1:, 0, 1] = 0.5 * (y_nk[:, :-1] + y_nk[:, 1:])
    S_nksv[:, :, 1, 1] = y_nk
    S_nksv[:, :-1, 2, 1] = 0.5 * (y_nk[:, :-1] + y_nk[:, 1:])
    S_nksv[:, -1, 2, 0] = np.nan
    return S_nksv.reshape((N * K, 3, 2))


bscalc = GPAW('mos2ws2.gpw')
bp = bscalc.atoms.cell.bandpath('GMKG', npoints=50)
x_k, xlabel_K, label_K = bp.get_linear_kpoint_axis()

eig_kn = []
color_kn = []
for wfs in bscalc.dft.ibzwfs:
    c_an = [(abs(P_ni)**2).sum(1) for P_ni in wfs.P_ani.values()]
    c_n = sum(c_an[:3]) / sum(c_an)
    color_kn.append(c_n)
    eig_kn.append(wfs.eig_n * Ha)

eigs = line_segments(x_k, np.array(eig_kn).T)
colors = np.array(color_kn).T.flatten()

fig, ax = plt.subplots()
lc = LineCollection(eigs)
lc.set_array(colors)
lines = ax.add_collection(lc)
fig.colorbar(lines)
ax.set_xlim(0, x_k[-1])
ax.set_ylim(-10, 0)
plt.show()
