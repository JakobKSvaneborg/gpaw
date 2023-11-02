# web-page: soc.png
from matplotlib.colors import Normalize
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
import numpy as np


def stereo_project_point(inpoint, axis=0, r=1):
    point = np.divide(inpoint * r, inpoint[axis] + r)
    point[axis] = 0
    return point


# Load data from nii2_soc.py
data = np.load('soc_data.npz')
theta, phi = data['theta'], data['phi']
soc = (data['soc'] - min(data['soc'])) * 10**3

# Convert angles to xyz coordinates
theta = theta * np.pi / 180
phi = phi * np.pi / 180
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)
points = np.array([x, y, z]).T

# Calculate stereographically project points
projected_points = []
for p in points:
    projected_points.append(stereo_project_point(p, axis=2))

fig, ax = plt.subplots(1, 1, figsize=(5 * 1.25, 5))

# Plot contour surface
norm = Normalize(vmin=min(soc), vmax=max(soc))
X, Y, Z = np.array(projected_points).T
xi = np.linspace(min(X), max(X), 100)
yi = np.linspace(min(Y), max(Y), 100)
zi = griddata((X, Y), soc, (xi[None, :], yi[:, None]))
ax.contour(xi, yi, zi, 15, linewidths=0.5, colors='k', norm=norm)
ax.contourf(xi, yi, zi, 15, cmap=plt.cm.plasma, norm=norm)

# Add additional contours
mask = np.argwhere(soc <= np.min(soc) + 0.05)
ax.scatter(X[mask], Y[mask], marker='o', c='midnightblue', s=5)
mask = np.argwhere(soc <= np.min(soc) + 0.001)
ax.scatter(X[mask], Y[mask], marker='o', c='k', s=5)
# Spin-orbit energy minimum
mask = np.argwhere(soc <= np.min(soc))
ax.scatter(X[mask], Y[mask], marker='o', c='white', s=5)
# z-axis direction
ax.scatter(X[0], Y[0], marker='o', c='k', s=10)

theta_min = round(theta[mask][0][0] * 180 / np.pi, 2)
phi_min = round(phi[mask][0][0] * 180 / np.pi, 2)
print(f'n = (theta, phi) = ({theta_min}, {phi_min})')

# Set plot details
ax.axis('equal')
ax.set_xlim(-1.05, 1.05)
ax.set_ylim(-1.05, 1.05)
ax.set_xticks([])
ax.set_yticks([])
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='plasma'), ax=ax)
cbar.ax.set_ylabel(r'$E_{soc} [meV]$')

# Save figure
plt.savefig('soc.png')
