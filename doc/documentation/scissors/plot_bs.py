import matplotlib.pyplot as plt
import numpy as np
from ase.units import Ha
from gpaw.new.ase_interface import GPAW
from matplotlib.collections import LineCollection


def line_segments(x_k: np.ndarray, y_nk: np.ndarray) -> np.ndarray:
    """Helper function for plotting colored bands.

    Converts (x,y) points to line segments.
    """
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


def plot(ibzwfs, bp, ax):
    x_k, xlabel_K, label_K = bp.get_linear_kpoint_axis()
    label_K = [label.replace('G', r'$\Gamma$') for label in label_K]

    eig_kn = []
    color_kn = []
    for wfs in ibzwfs:
        c_an = [(abs(P_ni)**2).sum(1) for P_ni in wfs.P_ani.values()]
        c_n = sum(c_an[:3]) / sum(c_an)
        color_kn.append(c_n)
        eig_kn.append(wfs.eig_n * Ha)

    eigs = line_segments(x_k, np.array(eig_kn).T)
    colors = np.array(color_kn).T.copy().flatten()
    lc = LineCollection(eigs)
    lc.set_array(colors)
    lines = ax.add_collection(lc)
    ax.set_xlim(0, x_k[-1])
    ax.set_ylim(-10, 0)
    ax.set_xticks(xlabel_K)
    ax.set_xticklabels(label_K)
    return lines


if __name__ == '__main__':
    fig, axs = plt.subplots(1, 3, sharey=True)
    for i, ax in enumerate(axs):
        bs_calc = GPAW(f'mos2ws2-{i}.gpw')
        bp = bs_calc.atoms.cell.bandpath('GMKG', npoints=50)
        lines = plot(bs_calc.dft.ibzwfs, bp, ax)
    cbar = fig.colorbar(lines)
    cbar.set_ticks(ticks=[0, 1], labels=['W', 'Mo'])
    # plt.show()
    plt.savefig('mos2ws2.png')
