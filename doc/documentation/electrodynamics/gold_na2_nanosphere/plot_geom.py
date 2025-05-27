# web-page: geom.png
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from ase.units import Bohr
from ase.visualize.plot import plot_atoms

from gpaw.tddft import TDDFT


# Initialize TDDFT and QSFDTD
td_calc = TDDFT('gs.gpw')


def generate_xygrid(d, g, box):

    vslice = 2  # yx

    # Determine the array lengths in each dimension
    ng = d.shape

    X = None
    Y = None
    U = None
    V = None

    # Slice data
    d_slice = np.rollaxis(d, vslice)[g[vslice], :, :]
    d_proj = np.zeros(d_slice.shape)
    for ind, val in np.ndenumerate(d_slice):
        d_proj[ind] = np.where(
            np.append(
                np.rollaxis(d, vslice)[:, ind[0], ind[1]], 1.0) != 0)[0][0]

    # Grids
    x = np.linspace(0, box[1], ng[1])
    y = np.linspace(0, box[0], ng[0])

    # Meshgrid and corresponding data
    X, Y = np.meshgrid(x, y)
    U = np.real(d_slice[1])  # y
    V = np.real(d_slice[0])  # x

    # Spacing
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    return d_slice, d_proj, (x, y, dx, dy), (X, Y, U, V)


poisson_solver = td_calc.hamiltonian.poisson
atoms = td_calc.atoms

box = np.diagonal(poisson_solver.cl.gd.cell_cv) * Bohr  # in Ang

# create figure
plt.figure(1, figsize=(4, 4))
plt.rcParams['font.size'] = 14

# prepare data
plotData = poisson_solver.classical_material.beta[0]
ng = plotData.shape

axis = 2
ax = plt.subplot(1, 1, 1)
g = [None, None, ng[2] // 2]

dmy1, d_proj, (x, y, dx, dy), dmy2 = generate_xygrid(plotData, g, box)

# choose the colourmap for the polarizable material here
plt.imshow(d_proj, interpolation='bicubic', origin='lower',
           cmap=ListedColormap(["goldenrod", "white"]),
           extent=[x[0] - dx / 2, x[-1] + dx / 2,
                   y[0] - dy / 2, y[-1] + dy / 2])

# Plot atoms
# switch x and y orientation for yx plot
pos = atoms.get_positions()
pos[:, [0, 1]] = pos[:, [1, 0]]
atoms.set_positions(pos)

cell = atoms.get_cell()
cell[:, [0, 1]] = cell[:, [1, 0]]
atoms.set_cell(cell)

# ASE plot atoms function
i, j = 1, 0

offset = np.array(
    [poisson_solver.qm.corner1[i],
     poisson_solver.qm.corner1[j]]) * 2 * Bohr

bbox = np.array(
    [poisson_solver.qm.corner1[i],
     poisson_solver.qm.corner1[j],
     poisson_solver.qm.corner2[i],
     poisson_solver.qm.corner2[j]]) * Bohr

plot_atoms(atoms, ax=None, show_unit_cell=2, offset=offset, bbox=bbox)

ax.autoscale()

# Classical grid
dmy1, dmy_proj, (x, y, dx, dy), dmy3 = generate_xygrid(plotData, g, box)
xx, yy = np.meshgrid(x, y)
plt.scatter(xx,
            yy,
            s=0.75, c='k', marker='o')

# Quantum grid
dmy1, dmy_proj, (x, y, dx, dy), dmy3 = generate_xygrid(
    plotData, g, box=np.diagonal(poisson_solver.qm.gd.cell_cv) * Bohr)
xx, yy = np.meshgrid(x, y)
plt.scatter(poisson_solver.qm.corner1[i] * Bohr + xx,
            poisson_solver.qm.corner1[j] * Bohr + yy,
            s=0.25, c='k', marker='o')

# Labels
plt.xlabel('y [Ang]')
plt.ylabel('x [Ang]')

# Plot
plt.tight_layout()
plt.savefig('geom.png')
