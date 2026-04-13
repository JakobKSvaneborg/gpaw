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
    * plots the BZ together with the cartesian positions of the VBM and CBM
      k-points.

Usage::

    python plot_vbm_cbm_bz.py my_calc.gpw [out.png]
"""
from __future__ import annotations

import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull, Voronoi

from gpaw import GPAW
from gpaw.spinorbit import soc_eigenstates


def build_first_bz(rec_cell: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Build the 1st Brillouin zone from reciprocal lattice vectors.

    Parameters
    ----------
    rec_cell
        3x3 array whose rows are the reciprocal lattice vectors
        (including the 2*pi factor), in 1/Å.

    Returns
    -------
    vertices
        (N, 3) array with the cartesian coordinates of the BZ vertices.
    faces
        List of (M_i, 3) arrays with the vertices of each face, ordered
        such that they form a closed polygon.
    """
    # Generate a small set of reciprocal lattice points around the origin;
    # the Voronoi cell surrounding the origin is the 1st Brillouin zone.
    N = 1
    grid = np.array([[i, j, k]
                     for i in range(-N, N + 1)
                     for j in range(-N, N + 1)
                     for k in range(-N, N + 1)])
    points = grid @ rec_cell

    vor = Voronoi(points)

    # Find the Voronoi region corresponding to the origin (index of the
    # all-zero row in ``grid``).
    origin_idx = np.where(np.all(grid == 0, axis=1))[0][0]
    region_idx = vor.point_region[origin_idx]
    region = vor.regions[region_idx]
    assert -1 not in region, 'BZ region is unbounded - increase N'

    vertices = vor.vertices[region]

    # Collect the faces of the BZ.  A Voronoi face is shared by the origin
    # and one neighbouring lattice point; it is the set of ridge vertices
    # of ridges that involve ``origin_idx``.
    faces = []
    for (p, q), ridge in zip(vor.ridge_points, vor.ridge_vertices):
        if origin_idx in (p, q) and -1 not in ridge:
            face = vor.vertices[ridge]
            # Order the vertices around the face so that Poly3DCollection
            # draws a proper polygon.
            faces.append(_order_polygon(face))

    return vertices, faces


def _order_polygon(verts: np.ndarray) -> np.ndarray:
    """Order the vertices of a planar polygon counter-clockwise."""
    centroid = verts.mean(axis=0)
    # Build an in-plane 2D basis.
    normal = np.cross(verts[1] - verts[0], verts[2] - verts[0])
    normal /= np.linalg.norm(normal)
    e1 = verts[0] - centroid
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    angles = np.array([np.arctan2((v - centroid) @ e2, (v - centroid) @ e1)
                       for v in verts])
    return verts[np.argsort(angles)]


def fold_to_first_bz(kpt_c: np.ndarray,
                     rec_cell: np.ndarray) -> np.ndarray:
    """Fold a cartesian k-point into the 1st Brillouin zone.

    Works by trying all neighbouring reciprocal lattice translations and
    picking the one that gives the smallest distance to the origin.
    """
    N = 1
    best = kpt_c
    best_norm = np.linalg.norm(kpt_c)
    for i in range(-N, N + 1):
        for j in range(-N, N + 1):
            for k in range(-N, N + 1):
                shift = np.array([i, j, k]) @ rec_cell
                candidate = kpt_c - shift
                n = np.linalg.norm(candidate)
                if n < best_norm - 1e-10:
                    best = candidate
                    best_norm = n
    return best


def main(gpw_file: str, out_file: str = 'vbm_cbm_bz.png') -> None:
    calc = GPAW(gpw_file, txt=None)

    # --- Reciprocal cell (rows are b1, b2, b3 in 1/Å, incl. 2π).
    rec_cell = 2 * np.pi * np.array(calc.atoms.cell.reciprocal())

    # --- Full BZ k-points in scaled (fractional) coordinates -> cartesian.
    kpts_scaled = calc.get_bz_k_points()  # shape (nbz, 3)
    kpts_cart = kpts_scaled @ rec_cell    # shape (nbz, 3), 1/Å

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

    # Fold into the 1st BZ (the k-point grid is usually in [0, 1), which
    # wraps out of the Voronoi cell for non-orthorhombic lattices).
    vbm_cart = fold_to_first_bz(kpts_cart[k_vbm], rec_cell)
    cbm_cart = fold_to_first_bz(kpts_cart[k_cbm], rec_cell)

    direct = np.allclose(vbm_cart, cbm_cart, atol=1e-6)
    gap = e_cbm - e_vbm

    print(f'Number of valence electrons: {nelec}')
    print(f'VBM: band {vbm_band}, k = {kpts_scaled[k_vbm]} '
          f'(scaled), E = {e_vbm:.4f} eV')
    print(f'CBM: band {cbm_band}, k = {kpts_scaled[k_cbm]} '
          f'(scaled), E = {e_cbm:.4f} eV')
    print(f'Band gap: {gap:.4f} eV '
          f'({"direct" if direct else "indirect"})')

    # --- Build the 1BZ and plot.
    _, faces = build_first_bz(rec_cell)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    poly = Poly3DCollection(faces, alpha=0.15, facecolor='tab:blue',
                            edgecolor='k', linewidths=1.0)
    ax.add_collection3d(poly)

    # Gamma point.
    ax.scatter([0], [0], [0], color='k', s=40)
    ax.text(0, 0, 0, r'  $\Gamma$', fontsize=12)

    ax.scatter(*vbm_cart, color='tab:red', s=120, depthshade=False,
               label=f'VBM ({e_vbm:.2f} eV)')
    ax.scatter(*cbm_cart, color='tab:green', s=120, depthshade=False,
               marker='^', label=f'CBM ({e_cbm:.2f} eV)')

    # Make axes equal so the BZ isn't distorted.
    all_verts = np.vstack([v for v in faces])
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
    print(f'Saved plot to {out_file}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    gpw = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else 'vbm_cbm_bz.png'
    main(gpw, out)
