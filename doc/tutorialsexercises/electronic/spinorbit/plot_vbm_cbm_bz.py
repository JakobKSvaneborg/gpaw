"""Plot the location of the VBM and CBM in the 1st Brillouin zone.

The script
    * loads a GPAW calculator from a .gpw file,
    * applies spin-orbit coupling with :func:`gpaw.spinorbit.soc_eigenstates`,
    * finds the valence-band maximum (VBM) and conduction-band minimum (CBM)
      from the total number of valence electrons (with SOC every band holds
      one electron, so band index ``nelec - 1`` is the VBM and ``nelec`` the
      CBM),
    * builds the 1st Brillouin zone as the Voronoi cell around the origin of
      the reciprocal lattice, and
    * plots the BZ together with the positions of the VBM and CBM k-points.

The dimensionality of the plot is chosen automatically from
``calc.atoms.pbc``: a 3D BZ is drawn for a bulk calculation (pbc = TTT) and
a 2D BZ polygon for a 2D material (exactly two periodic directions).

Usage::

    python plot_vbm_cbm_bz.py my_calc.gpw [out.png]
"""
from __future__ import annotations

import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Voronoi

from gpaw import GPAW
from gpaw.spinorbit import soc_eigenstates


# ---------------------------------------------------------------------------
# Brillouin zone construction
# ---------------------------------------------------------------------------

def build_first_bz_3d(rec_cell: np.ndarray
                      ) -> list[np.ndarray]:
    """Build the 3D 1st Brillouin zone as a list of face polygons.

    ``rec_cell`` is a 3x3 array whose rows are the reciprocal lattice vectors
    (including the 2*pi factor), in 1/Å.
    """
    N = 1
    grid = np.array([[i, j, k]
                     for i in range(-N, N + 1)
                     for j in range(-N, N + 1)
                     for k in range(-N, N + 1)])
    points = grid @ rec_cell

    vor = Voronoi(points)
    origin_idx = np.where(np.all(grid == 0, axis=1))[0][0]
    region = vor.regions[vor.point_region[origin_idx]]
    assert -1 not in region, 'BZ region is unbounded - increase N'

    faces = []
    for (p, q), ridge in zip(vor.ridge_points, vor.ridge_vertices):
        if origin_idx in (p, q) and -1 not in ridge:
            faces.append(_order_polygon_3d(vor.vertices[ridge]))
    return faces


def build_first_bz_2d(rec_cell_2d: np.ndarray) -> np.ndarray:
    """Build the 2D 1st Brillouin zone as a single CCW-ordered polygon.

    ``rec_cell_2d`` is a 2x2 array whose rows are the two reciprocal lattice
    vectors expressed in an in-plane orthonormal basis.
    """
    N = 1
    grid = np.array([[i, j]
                     for i in range(-N, N + 1)
                     for j in range(-N, N + 1)])
    points = grid @ rec_cell_2d

    vor = Voronoi(points)
    origin_idx = np.where(np.all(grid == 0, axis=1))[0][0]
    region = vor.regions[vor.point_region[origin_idx]]
    assert -1 not in region, 'BZ region is unbounded - increase N'

    verts = vor.vertices[region]
    centroid = verts.mean(axis=0)
    angles = np.arctan2(verts[:, 1] - centroid[1],
                        verts[:, 0] - centroid[0])
    return verts[np.argsort(angles)]


def _order_polygon_3d(verts: np.ndarray) -> np.ndarray:
    """Order the vertices of a planar 3D polygon counter-clockwise."""
    centroid = verts.mean(axis=0)
    normal = np.cross(verts[1] - verts[0], verts[2] - verts[0])
    normal /= np.linalg.norm(normal)
    e1 = verts[0] - centroid
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    angles = np.array([np.arctan2((v - centroid) @ e2, (v - centroid) @ e1)
                       for v in verts])
    return verts[np.argsort(angles)]


# ---------------------------------------------------------------------------
# Folding k-points into the 1st BZ
# ---------------------------------------------------------------------------

def fold_to_first_bz(kpt: np.ndarray,
                     rec_vecs: np.ndarray) -> np.ndarray:
    """Fold a k-point into the 1st BZ by trying nearest lattice translations.

    Works in any dimension: ``rec_vecs`` is a (d, d) matrix whose rows are
    the reciprocal lattice vectors in the chosen basis.
    """
    d = len(rec_vecs)
    best = kpt.copy()
    best_norm = np.linalg.norm(kpt)
    from itertools import product
    for shift_idx in product(range(-1, 2), repeat=d):
        shift = np.array(shift_idx) @ rec_vecs
        candidate = kpt - shift
        n = np.linalg.norm(candidate)
        if n < best_norm - 1e-10:
            best = candidate
            best_norm = n
    return best


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_bz_3d(faces: list[np.ndarray],
               vbm_cart: np.ndarray,
               cbm_cart: np.ndarray,
               e_vbm: float, e_cbm: float,
               gap: float, direct: bool,
               out_file: str) -> None:
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    poly = Poly3DCollection(faces, alpha=0.15, facecolor='tab:blue',
                            edgecolor='k', linewidths=1.0)
    ax.add_collection3d(poly)

    ax.scatter([0], [0], [0], color='k', s=40)
    ax.text(0, 0, 0, r'  $\Gamma$', fontsize=12)

    ax.scatter(*vbm_cart, color='tab:red', s=120, depthshade=False,
               label=f'VBM ({e_vbm:.2f} eV)')
    ax.scatter(*cbm_cart, color='tab:green', s=120, depthshade=False,
               marker='^', label=f'CBM ({e_cbm:.2f} eV)')

    all_verts = np.vstack(faces)
    span = np.abs(all_verts).max() * 1.1
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_zlim(-span, span)
    ax.set_box_aspect((1, 1, 1))

    ax.set_xlabel(r'$k_x$ (Å$^{-1}$)')
    ax.set_ylabel(r'$k_y$ (Å$^{-1}$)')
    ax.set_zlabel(r'$k_z$ (Å$^{-1}$)')
    ax.set_title(f'1st BZ with VBM/CBM  (gap = {gap:.3f} eV, '
                 f'{"direct" if direct else "indirect"})')
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)


def plot_bz_2d(poly_verts: np.ndarray,
               vbm_2d: np.ndarray,
               cbm_2d: np.ndarray,
               e_vbm: float, e_cbm: float,
               gap: float, direct: bool,
               out_file: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.add_patch(Polygon(poly_verts, closed=True, alpha=0.15,
                         facecolor='tab:blue', edgecolor='k', linewidth=1.2))

    ax.scatter([0], [0], color='k', s=40, zorder=3)
    ax.annotate(r'$\Gamma$', (0, 0), textcoords='offset points',
                xytext=(6, 6), fontsize=14)

    ax.scatter(*vbm_2d, color='tab:red', s=150, zorder=3,
               label=f'VBM ({e_vbm:.2f} eV)')
    ax.scatter(*cbm_2d, color='tab:green', s=150, zorder=3, marker='^',
               label=f'CBM ({e_cbm:.2f} eV)')

    span = np.abs(poly_verts).max() * 1.15
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)

    ax.set_xlabel(r'$k_1$ (Å$^{-1}$)')
    ax.set_ylabel(r'$k_2$ (Å$^{-1}$)')
    ax.set_title(f'1st BZ with VBM/CBM  (gap = {gap:.3f} eV, '
                 f'{"direct" if direct else "indirect"})')
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(gpw_file: str, out_file: str = 'vbm_cbm_bz.png') -> None:
    calc = GPAW(gpw_file, txt=None)
    atoms = calc.atoms

    # --- Reciprocal cell (rows are b1, b2, b3 in 1/Å, incl. 2π).
    rec_cell = 2 * np.pi * np.array(atoms.cell.reciprocal())

    # --- Full BZ k-points in scaled (fractional) coordinates -> cartesian.
    kpts_scaled = calc.get_bz_k_points()
    kpts_cart = kpts_scaled @ rec_cell

    # --- Spin-orbit coupled eigenvalues, shape (nbz, nbands_soc).
    soc = soc_eigenstates(calc)
    e_kn = soc.eigenvalues()

    # --- VBM / CBM band indices.  With SOC each band is singly occupied,
    # so the number of occupied bands equals the number of valence electrons.
    nelec = int(round(calc.get_number_of_electrons()))
    vbm_band = nelec - 1
    cbm_band = nelec
    if cbm_band >= e_kn.shape[1]:
        raise RuntimeError(
            f'Not enough bands available: need band {cbm_band}, '
            f'calculation has {e_kn.shape[1]}.')

    vbm_energies = e_kn[:, vbm_band]
    cbm_energies = e_kn[:, cbm_band]

    k_vbm = int(np.argmax(vbm_energies))
    k_cbm = int(np.argmin(cbm_energies))
    e_vbm = vbm_energies[k_vbm]
    e_cbm = cbm_energies[k_cbm]
    gap = e_cbm - e_vbm

    print(f'Number of valence electrons: {nelec}')
    print(f'pbc: {tuple(atoms.pbc)}')
    print(f'VBM: band {vbm_band}, k = {kpts_scaled[k_vbm]} '
          f'(scaled), E = {e_vbm:.4f} eV')
    print(f'CBM: band {cbm_band}, k = {kpts_scaled[k_cbm]} '
          f'(scaled), E = {e_cbm:.4f} eV')

    # --- Dispatch on dimensionality.
    periodic = np.where(atoms.pbc)[0]

    if len(periodic) == 3:
        vbm_pt = fold_to_first_bz(kpts_cart[k_vbm], rec_cell)
        cbm_pt = fold_to_first_bz(kpts_cart[k_cbm], rec_cell)
        direct = np.allclose(vbm_pt, cbm_pt, atol=1e-6)
        print(f'Band gap: {gap:.4f} eV '
              f'({"direct" if direct else "indirect"})')

        faces = build_first_bz_3d(rec_cell)
        plot_bz_3d(faces, vbm_pt, cbm_pt, e_vbm, e_cbm, gap, direct, out_file)

    elif len(periodic) == 2:
        # Reciprocal vectors of the two periodic directions (3D cartesian).
        b_pc = rec_cell[periodic]                     # shape (2, 3)

        # In-plane orthonormal basis built from b1 and b2.
        e1 = b_pc[0] / np.linalg.norm(b_pc[0])
        e2 = b_pc[1] - (b_pc[1] @ e1) * e1
        e2 /= np.linalg.norm(e2)
        basis = np.array([e1, e2])                    # shape (2, 3)

        # 2D coordinates of the reciprocal lattice and of the k-points.
        rec_cell_2d = b_pc @ basis.T                  # shape (2, 2)
        kpts_2d_all = kpts_cart @ basis.T             # shape (nbz, 2)

        vbm_pt = fold_to_first_bz(kpts_2d_all[k_vbm], rec_cell_2d)
        cbm_pt = fold_to_first_bz(kpts_2d_all[k_cbm], rec_cell_2d)
        direct = np.allclose(vbm_pt, cbm_pt, atol=1e-6)
        print(f'Band gap: {gap:.4f} eV '
              f'({"direct" if direct else "indirect"})')

        poly_verts = build_first_bz_2d(rec_cell_2d)
        plot_bz_2d(poly_verts, vbm_pt, cbm_pt,
                   e_vbm, e_cbm, gap, direct, out_file)

    else:
        raise NotImplementedError(
            f'Only 2D and 3D periodic systems are supported '
            f'(got pbc = {tuple(atoms.pbc)}).')

    print(f'Saved plot to {out_file}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    gpw = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else 'vbm_cbm_bz.png'
    main(gpw, out)
