# creates: f1.svg, f2.svg, f3.svg, f4.svg
import matplotlib.pyplot as p
p.show = lambda: None

import numpy as np
import matplotlib.pyplot as plt

# Some data
x = [1.4, 2.0, 3.0, 3.3]
y = [10, -1.1, 2.2, 5.0]
y2 = [1, 2, 3, 5]

plt.figure()   # Start a new figure
# Plot y versus x (called 'Series 1' in the legend).
#   b = blue.  o = show points.  - = show line.
plt.plot(x, y, 'bo-', label='Series 1')
# Plot y2 versus x (called 'Series 2' in the legend).
#   r = red.  x = show points as x'es.   -- = show dashed line
plt.plot(x, y2, 'rx--', label='Series 2')
plt.title('Title of the plot')
plt.xlabel('Label of the x-axis')
plt.ylabel('Label of the y-axis')
# Uncomment to set range of y values to show - similar for x:
# plt.ylim(0, 10)
plt.legend()  # make a legend, let matplotlib decide where to place it
plt.show()
plt.savefig('f1.svg')

plt.figure()
plt.plot(x, y, 'bo-', label='Series 1')
plt.plot(x, y2, 'rx--', label='Series 2')
plt.title('Title of the plot', size=24)
plt.xlabel('Label of the x-axis', size=16)
plt.ylabel('Label of the y-axis', size=16)
# Make tick marks larger, and increase the font.
plt.tick_params(axis='both', which='major', labelsize=16, size=10)
plt.ylim(-0.2, 10.2)     # Set range of y values to show - similar for x.
plt.legend(loc='upper left',
           fontsize=16)  # Make a legend, specify location.
plt.tight_layout()   # Fixes that otherwise some of the labels are cropped.
plt.show()
plt.savefig('f2.svg')

fig, axs = plt.subplots(1, 2, sharey=True)
x = np.linspace(0, 2 * np.pi, 100)
axs[0].plot(x, np.cos(x), label='cos')
axs[1].plot(x, np.sin(x), label='sin')
axs[0].legend()
axs[1].legend()
plt.show()
plt.savefig('f3.svg')

x = np.linspace(-1, 1, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2
N = 15
fig, ax = plt.subplots(1, 1)
ax.contour(X, Y, Z, N)
ax.set_aspect('equal')
plt.show()
plt.savefig('f4.svg')
