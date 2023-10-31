# web-page: Fe_site_sum_rules.png
"""Plot the pair site Zeeman energy calculated with a varrying number of bands.
"""

import numpy as np
import matplotlib.pyplot as plt

# Load data
rc_r = np.load('rc_r.npy')
EZ_r = np.load('EZ_r.npy')
sp_EZ_r = np.load('sp_EZ_r.npy')
EZ_nr = np.load('EZ_nr.npy')

# Plot data
fig, ax = plt.subplots()
ax.plot(rc_r, EZ_r)
ax.plot(rc_r, sp_EZ_r, c='0.5', linestyle=':')
for n, unocc in enumerate(4 * np.arange(9)):
    ax.plot(rc_r, EZ_nr[n].real, label=unocc)
ax.legend(title='unocc', loc=0)
ax.set_xlabel(r'$r_\mathrm{c}$ [$\mathrm{\AA}$]')
ax.set_xlim((0., 1.4))
ax.set_ylabel('Site Zeeman energy [eV]')
ax.set_ylim((0., None))
# Plot data on an inset displaying the flat region
ax2 = ax.inset_axes([0.565, 0.08, 0.41, 0.45])
ax2.plot(rc_r, EZ_r)
ax2.plot(rc_r, sp_EZ_r, c='0.5', linestyle=':')
for n, unocc in enumerate(4 * np.arange(9)):
    ax2.plot(rc_r, EZ_nr[n].real, label=unocc)
ax2.set_xlim((0.791, 1.365))
ax2.set_ylim((2.7, 2.8))

# Save as png
filename = 'Fe_site_sum_rules.png'
plt.savefig(filename, format='png', bbox_inches='tight')
