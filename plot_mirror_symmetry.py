from ase.units import Bohr
from gpaw import GPAW
from gpaw_symmetry_analysis import (
    get_transformed_coeffs,
    get_little_group_ops,
    analyze_band_structure_symmetry,
    calculate_representation_matrix,
)
import numpy as np
import matplotlib.pyplot as plt


def linspace3d(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size == 1:  # generate points in x-direction only
        assert b.size == 1
        x = np.linspace(a, b, n).reshape(-1, 1)
        y = np.zeros(x.shape)
        z = np.zeros(x.shape)
    else:
        ax, ay, az = a
        bx, by, bz = b

        x = np.linspace(ax, bx, n).reshape(-1, 1)
        y = np.linspace(ay, by, n).reshape(-1, 1)
        z = np.linspace(az, bz, n).reshape(-1, 1)
    return np.concatenate((x, y, z), axis=1)


folder = "/home/niflheim/amhso/Ti3AlC2/MoAlB/relaxed_sym_prim"
gs_gpw = "MoAlB_calc_surface_relaxed_and_symmetrized.gpw"
special_points_gpw = "MoAlB_calc_surface_special_sym_points.gpw"

calc = GPAW(f"{folder}/{special_points_gpw}")
atoms = calc.atoms
#         0  1  2  3  4   5   6   7    8
# kpts = [G, S, X, Y, XS, YS, GS, XS2, YS2]
kpts_kc = calc.get_bz_k_points()
bands = [78, 79, 80, 81]
u = 5
print(f"{kpts_kc = }")
print(f"{calc.get_number_of_bands() = }")
kpt_c = kpts_kc[u]
# psi = calc.get_pseudo_wave_function(band = bands[0], kpt = 8)
psit_nG = np.array(calc.wfs.kpt_u[u].psit_nG)[bands]
ops = get_little_group_ops(calc, kpt_c, symprec=1e-2)
print('---- All operations ----')
[print(op) for op in ops]
psi_rotated_nG = get_transformed_coeffs(
    calc, k_index=u, band_indices=bands, op_data=ops[3]
)
new_ops = []
for op in ops:
    R, t = op
    if R[2, 2] == 1:
        new_ops.append(op)
print('---- Reduced operations ----')
[print(op) for op in new_ops]

print('---- Analysis with full symmetry ----')
analyze_band_structure_symmetry(
    calc, k_point=kpt_c, band_range=(78, 81), debug=True, sym_ops=ops)


print('---- Analysis with reduced symmetry ----')
analyze_band_structure_symmetry(
    calc, k_point=kpt_c, band_range=(78, 81), debug=True, sym_ops=new_ops
)

psi_rotated_nG = get_transformed_coeffs(
    calc, k_index=u, band_indices=bands, op_data=new_ops[1]
)

# =====================================================================
# Coordinate mapping for this cell:
#
#   Grid axis 0  ->  lattice vector a1 = (0, 0, 3.1)   ->  Cartesian z
#   Grid axis 1  ->  lattice vector a2 = (3.21, 0, 0)  ->  Cartesian x
#   Grid axis 2  ->  lattice vector a3 = (0, 66.5, 0)  ->  Cartesian y
#
# gd.coords(c) returns 1D coordinates along the c-th lattice vector,
# which for this orthogonal cell equals one Cartesian component:
#   coords(0) = Cartesian z   (NOT x!)
#   coords(1) = Cartesian x   (the mirror direction)
#   coords(2) = Cartesian y   (out-of-plane)
# =====================================================================
cart_z = calc.wfs.gd.coords(0)  # grid axis 0: Cartesian z
cart_x = calc.wfs.gd.coords(1)  # grid axis 1: Cartesian x (mirror acts here)
cart_y = calc.wfs.gd.coords(2)  # grid axis 2: Cartesian y (out-of-plane)

psi78_xyz = calc.get_pseudo_wave_function(band=78, kpt=u)
psi79_xyz = calc.get_pseudo_wave_function(band=79, kpt=u)
psi80_xyz = calc.get_pseudo_wave_function(band=80, kpt=u)
psi81_xyz = calc.get_pseudo_wave_function(band=81, kpt=u)

plt.plot(cart_y, np.abs(psi78_xyz**2).sum((0, 1)), label=r'$\psi_1$')
plt.plot(cart_y, np.abs(psi79_xyz**2).sum((0, 1)), label=r"$\psi_1'$")
plt.plot(cart_y, np.abs(psi80_xyz**2).sum((0, 1)), label=r'$\psi_2$')
plt.plot(cart_y, np.abs(psi81_xyz**2).sum((0, 1)), label=r"$\psi_2'$")
plt.legend()
plt.savefig("XS.png")

mat1 = calculate_representation_matrix(
    calc, k_index=u, band_indices=[78, 79], op_data=new_ops[1]
)
print(f"{mat1 = }")
mat2 = calculate_representation_matrix(
    calc, k_index=u, band_indices=[80, 81], op_data=new_ops[1]
)
print(f"{mat2 = }")


# Shape: (N_gridaxis0, N_gridaxis1, N_gridaxis2) = (N_a1, N_a2, N_a3)
N_a1, N_a2, N_a3 = psi78_xyz.shape

print(f'{N_a1 = }')
print(f'{N_a2 = }')
print(f'{N_a3 = }')

i_a1 = N_a1 // 2                              # fix Cart. z at midpoint
i_a2 = N_a2 // 2                              # fix Cart. x at midpoint
i_a3 = np.argmin(np.abs(cart_y - 40))         # fix Cart. y ~ 40 Bohr

# --- Coarse-grid plot along Cart. x (grid axis 1, the mirror direction) ---
plt.figure()
plt.plot(cart_x * Bohr, np.abs(psi78_xyz[i_a1, :, i_a3].real),
         label=r'$\psi_1$')
plt.plot(cart_x * Bohr, np.abs(psi80_xyz[i_a1, :, i_a3].real),
         label=r'$\psi_2$')
plt.xlabel(r'$x_{\mathrm{cart}}$ / $\mathrm{\AA}$')
plt.legend()
plt.savefig('xline.png')

# --- Fine-grid reconstruction along Cart. x (grid axis 1) ---
psit78_G = calc.wfs.kpt_u[u].psit_nG[78]
psit80_G = calc.wfs.kpt_u[u].psit_nG[80]
pd = calc.wfs.pd
r_vxyz = calc.wfs.gd.get_grid_point_coordinates()
G_Gv = pd.get_reciprocal_vectors(q=u, add_q=True)
N_c = calc.wfs.gd.N_c

# Sweep grid axis 1 (Cart. x) over exactly one full period [0, a2).
# The evaluation positions and plotting positions must match exactly,
# otherwise the apparent mirror center drifts at k-points with fast
# Bloch oscillations.
Npoints = 400
cell_cv = calc.wfs.gd.cell_cv          # cell in Bohr, rows = lattice vectors
a2_v = cell_cv[1]                       # 2nd lattice vector (Cart. x direction)
La2 = np.linalg.norm(a2_v)             # |a2| in Bohr

r0_v = r_vxyz[:, i_a1, 0, i_a3]        # anchor: Cart. position at grid axis 1 = 0
frac = np.linspace(0, 1, Npoints, endpoint=False)  # fractional coords along a2
r_vx = r0_v[:, np.newaxis] + frac[np.newaxis, :] * a2_v[:, np.newaxis]
xfine = frac * La2                      # matching x-axis in Bohr

phase_Gx = np.exp(1j * G_Gv @ r_vx)
psi78_x = psit78_G @ phase_Gx / N_c.prod() / Bohr**(3 / 2)
psi80_x = psit80_G @ phase_Gx / N_c.prod() / Bohr**(3 / 2)

plt.figure()
plt.plot(xfine * Bohr, np.real(psi78_x),
         label=r'Re$(\psi_1)$', color='tab:blue')
plt.plot(xfine * Bohr, np.imag(psi78_x),
         label=r'Im$(\psi_1)$', ls='--', color='tab:blue')
plt.plot(xfine * Bohr, np.real(psi80_x),
         label=r'Re$(\psi_2)$', color='tab:orange')
plt.plot(xfine * Bohr, np.imag(psi80_x),
         label=r'Im$(\psi_2)$', ls='--', color='tab:orange')
plt.axvline(Bohr * xfine[np.argmin(np.abs(psi78_x))], color='k', ls='--')
plt.axvline(Bohr * xfine[np.argmin(np.abs(psi80_x))], color='k', ls='--')
plt.xlabel(r'$x_{\mathrm{cart}}$ / $\mathrm{\AA}$')
plt.legend()
plt.savefig('yline.png')

norm78 = np.linalg.norm(psi78_x)
norm80 = np.linalg.norm(psi80_x)
overlap = np.sum(psi78_x.conj() * psi80_x)

print(f'Norm 78: {norm78}')
print(f'Norm 80: {norm80}')
print(f'Overlap: {overlap}')
print(f'Normalized overlap = {np.abs(overlap) / (norm78 * norm80)}')
