# web-page: Fe_site_properties.png
"""Plot the site magnetization and site Zeeman energy of Fe(bcc)."""

import numpy as np
import matplotlib.pyplot as plt

# Load data
rc_r = np.load('Fe_rc_r.npy')
m_r = np.load('Fe_m_r.npy')
EZ_r = np.load('Fe_EZ_r.npy')
rc = np.load('Fe_rc.npy')
magmom = np.load('Fe_magmom.npy')

# Plot data
rlabel = r'$r_\mathrm{c}$ [$\mathrm{\AA}$]'
rlim = (0., 1.5)
plt.subplot(1, 2, 1)
plt.plot(rc_r, m_r)
plt.scatter([rc], [magmom], ec='k')
plt.axvline(rc, linestyle=':')
plt.xlabel(rlabel)
plt.xlim(rlim)
plt.ylabel(r'Site magnetization [$\mu_\mathrm{B}$]')
plt.ylim((0., None))
plt.subplot(1, 2, 2)
plt.plot(rc_r, EZ_r)
plt.axvline(rc, linestyle=':')
plt.xlabel(rlabel)
plt.xlim(rlim)
plt.ylabel('Site Zeeman energy [eV]')
plt.ylim((0., None))

# Save as png
filename = 'Fe_site_properties.png'
plt.tight_layout(w_pad=1.08)
plt.savefig(filename, format='png')
